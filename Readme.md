```bash
LLIE-EGCSO/
├── checkpoints/                   # 存放训练好的模型
│   ├── EdgeNet/                   # 存放边缘推测网络的模型权重
│   └── EGCSO/                     # 存放EGCSO模型的权重
├── datasets/                      # 存放数据集
│   ├── LOLv2/                     # 存放LOLv2数据集
│   │   ├── Real_captured/         # LOLv2 Real_captured 数据集
│   │   │   ├── Train/             # LOLv2 Real_captured 训练集
│   │   │   │   ├── Low/           # LOLv2 Real_captured 训练集低光图像
│   │   │   │   ├── Normal/        # LOLv2 Real_captured 训练集正常光照图像
│   │   │   │   └── Normal_edge/   # LOLv2 Real_captured 训练集正常光照图像的边缘图像
│   │   │   └── Test/              # LOLv2 Real_captured 测试集 
│   │   │       ├── Low/           # LOLv2 Real_captured 测试集低光图像
│   │   │       ├── Normal/        # LOLv2 Real_captured 测试集正常光照图像
│   │   │       └── Normal_edge/   # LOLv2 Real_captured 测试集正常光照图像的边缘图像
│   │   └── Synthetic/             # LOLv2 Synthetic 数据集
│   │       ├── Train/             # LOLv2 Synthetic 训练集
│   │       │   ├── Low/           # LOLv2 Synthetic 训练集低光图像
│   │       │   ├── Normal/        # LOLv2 Synthetic 训练集正常光照图像
│   │       │   └── Normal_edge/   # LOLv2 Synthetic 训练集正常光照图像的边缘图像
│   │       └── Test/              # LOLv2 Synthetic 测试集
│   │           ├── Low/           # LOLv2 Synthetic 测试集低光图像
│   │           ├── Normal/        # LOLv2 Synthetic 测试集正常光照图像
│   │           └── Normal_edge/   # LOLv2 Synthetic 测试集正常光照图像的边缘图像
│   ├── dataloader.py              # 数据加载类
│   ├── dataset_expansion.py       # 用于拓展边缘数据集的大小
│   ├── change.py                  # 用于将文件名中的 'normal' 替换为 'low'
│   ├── check_size.py              # 用于检查边缘图片的尺寸是否与数据集图片相匹配
│   ├── darkpixle_cal.py           # 用于计算数据集中的暗区像素占比
│   └── GT_mean.py                 # GT mean 操作
├── models/                        # 存放模型代码
│   ├── __init__.py 
│   ├── loss/                      # 损失函数代码
│   │   ├── __init__.py            
│   │   ├── loss_utils.py          # 损失函数模块代码
│   │   ├── lossfunction.py        # 损失函数代码 
│   │   └── vgg_arch.py            # VGG 模型代码
│   ├── module/
│   │   ├── attention.py           # 注意力模块代码
│   │   ├── BaseNet.py             # 简单的UNet网络
│   │   ├── block.py               # 定义一些基本的网络模块
│   │   ├── cbam.py                # cbam 注意力机制
│   │   ├── LBP.py                 # Lighten Back Projection 模块
│   │   ├── SPADE.py               # SPADE 模块
│   │   ├── EDADE.py               # EDADE 模块
│   │   └── transformer_util.py    # Transformer 模块
│   ├── ops/
│   │   └── iqa.py                 # 图像质量评估代码
│   ├── HVIT_ori.py                # 原始 HVIT 模型代码 （源自 CIDNet）
│   ├── HVIT.py                    # 改进的 HVIT 模型代码
│   ├── LCA.py                     # LCA 模块代码
│   ├── EGCSO.py                   # EGCSO 模型代码
│   └── EIN.py                     # EIN 模型代码
├── train/                         # 训练代码
│   ├── __init__.py                
│   ├── config.json                # 配置文件
│   ├── train_EIN.py               # EIN模型训练代码
│   └── train_EGCSO.py             # EGCSO模型训练代码
└── predict/                       # 预测代码
    ├── __init__.py               
    ├── inference.py               # 推理代码
    ├── predict_EGCSO.py           # EGCSO模型的预测代码
    ├── predict_EIN_result.py      # EIN模型的预测代码
    └── predict_EIN_show.py        # EIN模型的可视化代码
```
