import numpy as np
import torch as th
from diffusion import GaussianDiffusion
from noise_schedule import noise_related_calculate

# DDIM 的核心之一，从原始的时间步中，智能地间隔取出一定数量的时间步。
def space_timesteps(num_timesteps, section_counts):
    """
    num_timesteps: 原始过程中要分割的扩散步骤数。
    section_counts: 一个数字列表，或是包含逗号分隔数字的字符串。
                    特例: ddimN 的情况下,  N 是 DDIM 论文中的步长。
    :return: 更新后的一组扩散时间步。
    """
    # 如果传递的 section_counts 是一个字符串
    if isinstance(section_counts, str):
        # 1. 并且这个字符串以 ddim 开头
        if section_counts.startswith("ddim"):
            # 1.1 取出 ddim 后的数值，并转换为 int
            desired_count = int(section_counts[len("ddim") :])
            # 1.2 尝试通过 等间距取样（stride = i）的方法，找到合适的步长，
            # 使得选出来刚好是 desired_count 个时间步
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            # 1.3 如果找不到恰好对得上的整数间隔，就会报错
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        # 2. 非 DDIM，明确分段策略。假设传入的是 "10,10,10"，得到一个 [10, 10, 10] 的列表
        section_counts = [int(x) for x in section_counts.split(",")]
    
    # 2.1 列表 section_counts 的长度为 3，意味着把 timesteps 分为 3 段
    # // 表示整除，小数点后舍弃； % 表示取余，在除不尽时，尽量保证分段步长的均匀
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0       # 每个分段的起始点
    all_steps = []      # 所有等距离点的列表

    # 2.2 在每一段中，均匀地选出 section_count 个点
    for i, section_count in enumerate(section_counts):
        # 假设 1000 步，3段，size 就是 334，333，333
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            ) 
        # 在 [0, size - 1] 的区间上，取出 section_count 个等距离的点
        # 这些点之间的间隔数量为 section_count - 1；间隔小于 1 时取 1。
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        
        cur_idx = 0.0  # 当前循环中的等距离点
        taken_steps = []  # 当前循环中的等距离点的列表
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))  # round 四舍五入
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    
    return set(all_steps)  # 去重并返回


class SpacedDiffusion(GaussianDiffusion):
    """
    可跳过基础扩散过程步骤的扩散过程。

    :param use_timesteps: 要保留的原始扩散过程的时间步集合（序列或集合）。
    :param kwargs: kwargs 来创建基础扩散进程。
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)  # 修改后的时间步集合
        self.timestep_map = []  # 映射表
        self.original_num_steps = len(kwargs["betas"])  # 原始的扩散步数

        beta_related = noise_related_calculate(kwargs["betas"])
        last_alpha_cumprod = 1.0
        new_betas = []  # 构造新的跳跃 betas 序列

        # 使用 alphas_cumprod 来推导新的 betas 序列，确保每一步的信号变化与整体一致
        for i, alpha_cumprod in enumerate(beta_related["alphas_cumprod"]):
            if i in self.use_timesteps:
                # 假设从 1000 步缩减到了 250 步，间隔为 4，这段计算大致是：
                # beta_0 = 1 - (alphas_cumprod[t4] / alphas_cumprod[t0])
                # beta_1 = 1 - (alphas_cumprod[t8] / alphas_cumprod[t4])
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)  # new_betas 序列替换 kwargs["betas"]

        # 使用新的参数 kwargs 完成父类 GaussianDiffusion 的初始化
        super().__init__(**kwargs)  

    # 在调用父类的训练或采样接口前，先把模型包装一遍
    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    # 模型包装器，构造并返回 _WrappedModel 对象。目的是为了改写输入的时间步。
    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        # ts 是当前的稀疏时间步，例如 [0, 1, 2, ..., 49]
        # self.timestep_map 记录的是原始时间步，比如 [0, 20, 40, ..., 980]
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]

        # 有些模型默认 timestep 是 [0, 1000] 之间的 float；
        # 如果原始 diffusion 用了非 1000 步的计划，就会 rescale 一下。
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)

        # 现在模型输入的是原始时间步 t，一切恢复正常
        return self.model(x, new_ts, **kwargs)
