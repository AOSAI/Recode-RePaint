## 1. 代码复现-RePaint

## 2. 环境配置

- system: win11
- conda 虚拟环境: python=3.12
- pytorch 版本: 2.4.1
- 其它相关依赖：查看 requirements.txt, 或代码中查看导入包的缺失

## 3. 目录结构

```py
Recode-RePaint/
├── diffusionModel/           # 扩散模型核心(算法逻辑)
│   ├── noise_schedule.py       # beta 噪声时间表、相关计算
│   ├── diffusion.py            # 训练、原始采样、DDIM采样
│   ├── losses.py               # 损失函数的相关计算
│   └── utils.py                # 辅助函数
├── networkModel/             # 网络模型架构(神经网络)
│   ├── unet.py                 # U-Net + Resblock + Attention
│   ├── utils.py                # U-Net 相关的函数调用
│   └── fp16_util.py            # 暂未优化
├── tools/                    # 项目级工具
│   ├── logger.py               # 训练日志 setup
│   └── script_util.py          # ...
├── scripts/                  # 训练采样脚本入口
│   ├── image_train.py          # 训练模型
│   └── image_sample.py         # 采样生成
├── public/                   # 公共资源
│   ├── configs/                # 参数配置文件
│   └── docsImg/                # markdown所需图像
│   └── documents/*.md          # 每个文件的讲解笔记《小白指南》
│   └── jupyter/                # jupyter notebook
├── main.py                   # 运行入口
```

## 参考文献
