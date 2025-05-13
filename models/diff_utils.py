import torch as th
import numpy as np
import enum
import math


# --------------- 均值、方差、损失的类型 ---------------
class ModelMeanType(enum.Enum):
    """决定模型预测的内容是什么，用于训练"""
    PREVIOUS_X = enum.auto()    # 预测 x_{t-1}
    START_X = enum.auto()       # 预测 x_0
    EPSILON = enum.auto()       # 预测噪声 epsilon    

class ModelVarType(enum.Enum):
    """决定反向过程中协方差的计算方式，用于采样而非训练"""
    LEARNED = enum.auto()       # 可学习方差，无范围限制
    LEARNED_RANGE = enum.auto() # 可学习方差，线性映射到 [xx, xx] 区间
    FIXED_SMALL = enum.auto()   # 固定方差，posterior_variance，训练稳定，多样性低
    FIXED_LARGE = enum.auto()   # 固定方差，betas，多样性更高

class LossType(enum.Enum):
    """模型的损失计算方式"""
    MSE = enum.auto()       # 使用原始 MSE 损失（学习方差时使用 KL）
    RESCALED_MSE = enum.auto()  # 使用原始 MSE 损失（学习方差时使用 RESCALED_KL）
    KL = enum.auto()        # 使用变分下界 (variational lower-bound)
    RESCALED_KL = enum.auto()   # 与 KL 相似，但要重新缩放以估算整个 VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


# --------------- 转化张量、对齐形状 ---------------
def extract(arr, timesteps, broadcast_shape):
    """
    从 1D numpy 数组中按时间步采样, 转化成tensor, 并 broadcast 到目标形状。

    :param arr: 1D numpy 数组 (如 beta schedule)
    :param timesteps: shape 为 [batch_size] 的时间步张量。
    :param broadcast_shape: 用于广播的目标形状（如 [B, 1, 1, 1]）。
    :return: 已广播的张量, shape 为 broadcast_shape。
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# --------------- KL 散度计算 ---------------
def normal_kl(mean1, logvar1, mean2, logvar2):
    """ 两个对角高斯分布之间的 KL 散度计算, 对齐参数类型 Tensor (防御工程) """
    assert all(isinstance(x, th.Tensor) for x in (mean1, logvar1, mean2, logvar2))

    return 0.5 * (
        -1.0                                        # -1
        + logvar2 - logvar1                         # log(sigma2^2 / sigma1^2)
        + th.exp(logvar1 - logvar2)                 # sigma1^2 / sigma2^2
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2) # (mu1 - mu2)^2 / sigma2^2
    )

def mean_flat(tensor):
    """ 把每个样本的 KL 散度结果，沿着非 batch 维度做平均 """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# --------------- 计算负对数似然 ---------------
def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    计算离散到给定图像的高斯分布的对数似然值

    :param x: 目标图像。假定是 uint8 值，并已将其调整为 [-1, 1] 范围
    :param means: 高斯平均张量
    :param log_scales: 高斯对数 stddev 张量。
    :return: 一个类似 x 的对数概率张量（单位：纳特）
    """
    assert x.shape == means.shape == log_scales.shape

    # 中心化并标准化输入
    centered_x = x - means          #  x - μ
    inv_stdv = th.exp(-log_scales)  #  1 / σ

    # 求两个近邻的 CDF 值
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)       # (x - μ + Δ/2) / σ
    cdf_plus = 0.5 * (1.0 + th.erf(plus_in / math.sqrt(2))) # Φ(plus_in) 
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)        # (x - μ - Δ/2) / σ
    cdf_min = 0.5 * (1.0 + th.erf(min_in / math.sqrt(2)))   # Φ(min_in)

    # 边界情况分割处理（x ≈ -1 or x ≈ +1）
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))                    
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))    
    cdf_delta = cdf_plus - cdf_min

    # 使用 torch.where 合并三种情况
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,   # 只保留左边边界的概率
        th.where(
            x > 0.999, 
            log_one_minus_cdf_min,  # 只保留右边边界的概率
            th.log(cdf_delta.clamp(min=1e-12))  # 中间区间，确保 log() 不会出错
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs