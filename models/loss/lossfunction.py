import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from pytorch_msssim import ssim
from models.loss.vgg_arch import VGGFeatureExtractor, Registry
from models.loss.loss_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# hed_loss2是对hed_lossORI的改进，hed_lossORI是原始的hed损失函数
def hed_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.1).float()).float()
    num_negative = torch.sum((mask <= 0.).float()).float()

    mask[mask > 0.1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)
    inputs = torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), targets.float())

    return l_weight * torch.sum(cost)


# bdcn_loss2是对bdcn_lossORI的改进，bdcn_lossORI是原始的bdcn损失函数
def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float()  # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float()  # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative)  # 0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    inputs = torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    # cost = torch.mean(cost.float().mean((1, 2, 3))) # before sum
    cost = torch.sum(cost.float().mean((1, 2, 3)))  # before sum

    return l_weight * cost


# bdcn_lossORI是原始的bdcn损失函数
def bdcn_lossORI(inputs, targets, l_weights=1.1, cuda=False):
    """
    :param l_weights: 1.1
    :param cuda: False
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    # print(cuda)
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * 1.1 / valid  # balance = 1.1
    weights = torch.Tensor(weights)
    # if cuda:
    weights = weights.cuda()
    inputs = torch.sigmoid(inputs)
    loss = torch.nn.BCELoss(weights, reduction='sum')(inputs.float(), targets.float())

    return l_weights * loss


def BCE_loss(inputs, targets, l_weight=1.3):
    # 计算正边缘和负边缘的数量
    targets = targets.long()
    mask = (targets > 0.1).float()
    num_positive = torch.sum((mask > 0.0).float()).float()  # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float()  # <= 0.1

    # 计算权重 alpha 和 beta
    alpha = num_negative / (num_positive + num_negative)  # 权重 alpha
    beta = num_positive / (num_positive + num_negative)   # 权重 beta

    # 计算交叉熵损失
    inputs = torch.sigmoid(inputs)
    mask[mask > 0.] = alpha
    mask[mask <= 0.] = beta * 1.1
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())

    cost = torch.sum(cost.float().mean((1, 2, 3)))  # before sum
    return l_weight * cost


# rcf_loss是原始的rcf损失函数
def rcf_loss(inputs, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask > 0.5).float()).float()  # ==1.
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0.
    inputs = torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), label.float())

    return 1. * torch.sum(cost)


# cats_loss是原始的cats损失函数
def bdrloss(prediction, label, radius, device='cpu'):
    """
    The boundary tracing loss that handles the confusing pixels.
    :param prediction:
    :param label:
    :param radius:
    :param device: str
    """
    filt = torch.ones(1, 1, 2 * radius + 1, 2 * radius + 1)
    filt.requires_grad = False
    filt = filt.to(device)

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)
    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1 - label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return cost.sum()


# textureloss是原始的texture损失函数
def textureloss(prediction, label, mask_radius, device='cpu'):
    """
    The texture suppression loss that smooths the texture regions.
    """
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.to(device)
    filt2 = torch.ones(1, 1, 2 * mask_radius + 1, 2 * mask_radius + 1)
    filt2.requires_grad = False
    filt2 = filt2.to(device)

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1 - pred_sums / 9, 1e-10, 1 - 1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss)


def cats_loss(prediction, label, l_weight=None, device='cpu'):
    # tracingLoss
    if l_weight is None:
        l_weight = [0., 0.]
    tex_factor, bdr_factor = l_weight
    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0
    prediction = torch.sigmoid(prediction)
    # print('bce')
    cost = torch.sum(torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduce=False))
    label_w = (label != 0).float()
    # print('tex')
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4, device=device)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4, device=device)

    return cost + bdr_factor * bdrcost + tex_factor * textcost


# 休伯损失+感知损失+SSIM损失
class CombinedLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.huber_loss = nn.SmoothL1Loss()
        self.vgg = vgg16(pretrained=True).features[:16].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def perceptual_loss(self, pred, target):
        pred_feature = self.vgg(pred)
        target_feature = self.vgg(target)
        return F.mse_loss(pred_feature, target_feature)

    def forward(self, pred, target):
        huber_loss = self.huber_loss(pred, target)
        perceptual_loss = self.perceptual_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        total_loss = self.alpha * huber_loss + self.beta * perceptual_loss + self.gamma * ssim_loss
        return total_loss


# 休伯损失+SSIM损失
class HuberSSIMLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(HuberSSIMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.huber_loss = nn.SmoothL1Loss()

    def forward(self, pred, target):
        huber_loss = self.huber_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        total_loss = self.alpha * huber_loss + self.beta * ssim_loss
        return total_loss


# L1损失+SSIM损失
class L1SSIMLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(L1SSIMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        total_loss = self.alpha * l1_loss + self.beta * ssim_loss
        return total_loss


# L2损失+感知损失
class L2PerceptualLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(L2PerceptualLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l2_loss = nn.MSELoss()
        self.vgg = vgg16(pretrained=True).features[:16].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def perceptual_loss(self, pred, target):
        pred_feature = self.vgg(pred)
        target_feature = self.vgg(target)
        return F.mse_loss(pred_feature, target_feature)

    def forward(self, pred, target):
        l2_loss = self.l2_loss(pred, target)
        perceptual_loss = self.perceptual_loss(pred, target)
        total_loss = self.alpha * l2_loss + self.beta * perceptual_loss
        return total_loss


class LuminanceAdjustmentLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, epsilon=1e-6):
        super(LuminanceAdjustmentLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, low, high, estimated):
        # Calculate the squared difference loss between low and high texture
        L1_loss = torch.mean(torch.pow(low / (estimated + self.epsilon) - high, 2))

        gradient_x = torch.abs(torch.gradient(estimated, dim=2)[0])
        gradient_y = torch.abs(torch.gradient(estimated, dim=3)[0])
        gradient_weight = 1.0 / (gradient_x + gradient_y + self.epsilon)

        # Calculate the piece-wise smoothness constraint loss
        smoothness_loss = torch.mean(torch.abs(gradient_weight * torch.gradient(estimated, dim=(2, 3))[0]))

        # Calculate the gradient similarity loss
        similarity_loss = torch.mean(
            torch.abs(torch.gradient(low, dim=(2, 3))[0] - torch.gradient(estimated, dim=(2, 3))[0]))

        # Calculate the total loss
        total_loss = L1_loss + self.alpha * smoothness_loss + self.beta * similarity_loss

        return total_loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class SSIMTVLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.001):
        super(SSIMTVLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.tv_loss = TVLoss()

    def forward(self, pred, target):
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        tv_loss = self.tv_loss(pred)
        loss = self.alpha * ssim_loss + self.beta * tv_loss
        return loss


class SmoothMseLoss(nn.Module):
    def __init__(self, smooth_weight=0.1):
        super(SmoothMseLoss, self).__init__()
        self.smooth_weight = smooth_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse_loss(y_pred, y_true)
        smooth_loss = self.smoothness_loss(y_pred)
        total_loss = mse_loss + self.smooth_weight * smooth_loss
        return total_loss

    @staticmethod
    def smoothness_loss(y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        return torch.mean(dx) + torch.mean(dy)


_reduction_modes = ['none', 'mean', 'sum']


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1).cuda()

        self.weight = loss_weight

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss * self.weight


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward output.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight
