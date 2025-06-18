from abc import ABC, abstractmethod
import numpy as np
import torch as th


def create_named_schedule_sampler(name, diffusion):
    """ 从预定义采样器库中创建 ScheduleSampler """
    
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    扩散过程中各时间步的分布，旨在减少目标的方差。
    默认情况下，采样器执行 “无偏重要度采样”，即目标平均值保持不变。
    不过，子类可以覆盖 sample() 来改变重新采样项的加权方式，从而允许目标的实际变化。
    """

    @abstractmethod
    def weights(self):
        """ 获取权重的 numpy 数组，每个扩散步骤一个。权重无需标准化，但必须为正值 """

    def sample(self, batch_size, device):
        """ 批次的重要性采样时间步 """
        w = self.weights()  # 权重向量，UniformSampler 中的 _weights
        p = w / np.sum(w)   # 概率分布，每个 t 被采样的概率 q[t] = 1/T、

        # 从 T 个时间步中采 batch_size 个 index，按照 q 分布采样
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)  # 从 q(t) 采样的 t
        indices = th.from_numpy(indices_np).long().to(device)

        # 对每个被采样到的 index，我们要计算一个“反权重”（用于 loss 加权）
        weights_np = 1 / (len(p) * p[indices_np])   # 1 / (T * q(t))
        weights = th.from_numpy(weights_np).float().to(device)

        # indices 表示新的 timesteps，weights 表示loss的权重
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        # 均匀的，使每一个时间步 t 的权重都为1
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossSecondMomentResampler(ScheduleSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        
        # 为每个 timestep 记录最近 history_per_term 个 loss，构成一个滑动窗口
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        # 用来记录每个 timestep 已经积累了多少个 loss 样本
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=int)

    def _warmed_up(self):
        # 如果所有 timestep 都积满了 loss 历史，返回 true
        return (self._loss_counts == self.history_per_term).all()

    def weights(self):
        # 满足条件则使用加权采样，否则全部用均匀采样（冷启动策略）
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        
        # 重要性度量
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # 如果当前 timestep 的历史记录已满，则替换最旧一项；
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                # 否则就在后面添加新 loss
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

