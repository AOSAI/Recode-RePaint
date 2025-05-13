import math
import numpy as np

def linear_beta_schedule(timesteps, use_scale):
    """ 1000 步的原始 beta 线性时间表。包含步数缩放机制。"""
    
    scale = 1000 / timesteps if use_scale else 1
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, max_beta=0.999):
    """ beta 余弦时间表。构造更平滑的、收敛更好的噪声序列"""
    
    betas = []
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        alpha_bar1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2 
        alpha_bar2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2 
        betas.append(min(1 - alpha_bar2 / alpha_bar1, max_beta))
    return np.array(betas)

def get_noise_schedule(schedule_name, timesteps, use_scale=True):
    """ 根据不同的方案构造 beta 时间表"""
    
    if schedule_name == "linear":
        betas = linear_beta_schedule(timesteps, use_scale)
    elif schedule_name == "cosine":
        betas = cosine_beta_schedule(timesteps)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    
    return betas

def noise_related_calculate(betas):
    """ 根据 beta 构造 alpha, alpha_bar 等时间相关参数"""
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)

    # 后验方差
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    # 后验对数方差。由于扩散链开始时的后验方差为 0，因此对数计算被剪切。
    posterior_log_variance_clipped = np.log(
        np.append(posterior_variance[1], posterior_variance[1:])
    )
    # 后验均值的两个系数
    posterior_mean_coef1 = (
        betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
    )

    return {
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "alphas_cumprod_next": alphas_cumprod_next,
        "sqrt_alphas_cumprod": np.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": np.sqrt(1.0 - alphas_cumprod),
        "log_one_minus_alphas_cumprod": np.log(1.0 - alphas_cumprod),
        "sqrt_recip_alphas_cumprod": np.sqrt(1.0 / alphas_cumprod),
        "sqrt_recipm1_alphas_cumprod": np.sqrt(1.0 / alphas_cumprod - 1),
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }