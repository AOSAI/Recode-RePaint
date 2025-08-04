import blobfile as bf
import numpy as np
import torch as th
import torch.nn as nn
from torch.optim import AdamW
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import math
import copy, re
import functools
from tools import logger
from models.resample import LossSecondMomentResampler, UniformSampler

# 对于 ImageNet 实验，这是一个很好的默认值。
# 我们发现，在最初的 ~1K 步训练中，lg_loss_scale 迅速攀升至 20-21
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
        self, *, model, diffusion,
        data, batch_size, microbatch, lr, ema_rate,
        log_interval, save_interval, resume_checkpoint,
        use_fp16=False, fp16_scale_growth=1e-3,
        schedule_sampler=None, weight_decay=0.0, lr_anneal_steps=0,
    ):
        # ------------ 1. 传入的参数，处理并全局化 ------------
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        # ------------ 2. 训练控制逻辑的状态变量初始化 ------------
        self.step = 0
        self.resume_step = 0
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.device = next(self.model.parameters()).device

        # ------------ 3. 断点恢复检查、FP16 精度、构造参数优化器 ------------
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        # ------------ 4. 若断点存在，加载对应的 opt、ema 参数；否则构造 ema ------------
        if self.resume_step:
            self._load_optimizer_state()
            params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
            self.ema_params = params
        else:
            params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]
            self.ema_params = params


    # 3.1 检查是否有训练好的模型文件, 有的话启用断点恢复机制
    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(th.load(resume_checkpoint, map_location=self.device))

    # 3.2 启用 Float16 精度训练时的处理
    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    # 4.1 从断点文件中加载优化器状态（即 AdamW 中的动量、方差等）
    def _load_optimizer_state(self):
        opt_checkpoint = bf.join(
            bf.dirname(self.resume_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location=self.device)
            self.opt.load_state_dict(state_dict)

    # 4.2 从断点文件中加载指定 EMA 衰减率（rate）对应的参数
    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)
        filename = f"ema_{rate}_{self.resume_step:06d}.pt"
        ema_checkpoint = bf.join(bf.dirname(self.resume_checkpoint), filename)
        
        if bf.exists(ema_checkpoint):
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load(ema_checkpoint, map_location=self.device)
            ema_params = self._state_dict_to_master_params(state_dict)
        return ema_params

    # ------------ 5. 训练的主循环函数 ------------
    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)  # 取自 image_datasets
            self.run_step(batch, cond)     # 训练核心

            # 定期打印日志；定期保存模型、优化器、EMA状态
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            
            self.step += 1
        
        # 当训练结束但没到 save_interval，就手动补存一次 checkpoint
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    # 5.1 训练核心（前向传播、反向传播、优化器更新、精度调整）
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    # 5.1.1 前向传播、反向传播
    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)  # 梯度清零
        
        for i in range(0, batch.shape[0], self.microbatch):
            # batch 被拆成多个迷你 batch；无条件 & 有条件
            micro = batch[i : i + self.microbatch].to(self.device)
            micro_cond = {
                k: v[i : i + self.microbatch].to(self.device)
                for k, v in cond.items()
            }

            # 重要性采样（resample）返回的新的时间步和权重
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            # 封装损失函数 training_losses，并绑定第二行的参数
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model, micro, t, model_kwargs=micro_cond,
            )
            losses = compute_losses()  # 实际执行损失函数

            # 如果重要性采样使用了 loss-second-moment，更新权重（单卡版本）
            if isinstance(self.schedule_sampler, LossSecondMomentResampler):
                self.schedule_sampler.update_with_all_losses(
                    t, losses["loss"].detach()
                )

            # 逐元素乘上采样权重，做加权平均；日志记录每个 loss 项
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            # FP16 下需要放大 loss，避免梯度变 0；否则直接反向传播
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    # 5.1.2 优化器更新、精度调整
    def optimize_fp16(self):
        # 核心步骤
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return
        
        # 把模型中 FP16 的梯度拷贝给 FP32 的 master_params；并缩放回原始数值范围
        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        
        # 把更新后的 FP32 master 参数拷贝回 FP16 模型参数中；尝试提高 loss scale
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    # 5.1.3 优化器更新
    def optimize_normal(self):
        # 记录梯度范数；学习率动态调整；真正地执行参数更新
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()

        # 对 FP32 master 参数执行 EMA 滑动平均
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        """ 计算当前所有参数的梯度范数 (L2 范数) 并记录日志 """
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    # def _anneal_lr(self):
    #     """ 学习率 lr 余弦退火 (Cosine Annealing) """
    #     if not self.lr_anneal_steps:
    #         return

    #     frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
    #     frac_done = min(frac_done, 1.0)  # 防止超过1

    #     # 设置最小学习率，可调节
    #     min_lr = self.lr * 0.0  # 最小为初始lr的
    #     cosine_factor = 0.5 * (1 + math.cos(math.pi * frac_done))
    #     lr = min_lr + (self.lr - min_lr) * cosine_factor

    #     for param_group in self.opt.param_groups:
    #         param_group["lr"] = lr

    def _anneal_lr(self):
        """ 学习率 lr 线性退火 (Linear Annealing) """
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    # 5.1.4 核心训练状态的日志数据记录
    def log_step(self):
        # step 表示主循环调用过的次数（包括从断点恢复时的 resume_step）
        logger.logkv("step", self.step + self.resume_step)
        # samples 表示目前为止训练过的总样本数（samples processed）；
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.batch_size)
        if self.use_fp16:
            # 如果启用 FP16 计算，记录当前的 loss scale（以 log2 的形式）
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    # 5.2 模型、优化器、EMA状态的保存
    def save(self):
        # 内部函数，用于保存单个模型参数版本
        def save_checkpoint(rate, params, total_step):
            # 将主参数或 EMA 参数，转换为 PyTorch 的 state_dict 格式
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            # 如果不是 EMA (没有 rate)，保存主模型，否则保存 EMA 参数
            if not rate:
                filename = f"model{(total_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(total_step):06d}.pt"
            with bf.BlobFile(bf.join(logger.get_dir(), filename), "wb") as f:
                th.save(state_dict, f)

        total_step = self.step + self.resume_step
        # 实际调用内部函数，先保存 model，再逐个保存 EMA
        save_checkpoint(0, self.master_params, total_step)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params, total_step)

        # 保存优化器状态 opt
        opt_file = bf.join(logger.get_dir(), f"opt{(total_step):06d}.pt")
        with bf.BlobFile(opt_file , "wb") as f:
            th.save(self.opt.state_dict(), f)

    # 5.2.1 参数格式转换为 PyTorch 的 state_dict
    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        # 获取当前模型的 state_dict，包含所有参数名和对应 tensor
        state_dict = self.model.state_dict()
        # 遍历模型的所有参数，将 master_params[i] 依次替换到 state_dict[name]
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    # 4.2.1 将加载的 state_dict（通常来自 EMA 检查点）转为 master 参数格式
    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


# 提取预训练模型文件名中的数字, 作为 resume_step 返回
def parse_resume_step_from_filename(filename):
    match = re.search(r'model(\d+)\.pt$', filename)
    if match:
        return int(match.group(1))
    return 0


# 不打印日志，只是将 loss 信息注册到 logger 里，供主循环处理
def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


# 使用指数移动平均法更新目标参数 target, 使其更接近源参数 source
def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


# ------------ Helpers to train with 16-bit precision. ------------
def make_master_params(model_params):
    """ 将模型参数复制到全精度参数的（不同形状的）列表中 """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def model_grads_to_master_grads(model_params, master_params):
    """ 将模型参数中的梯度复制到 self.master_params 中的主参数中 """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )

def master_params_to_model_params(model_params, master_params):
    """ 将主参数数据拷贝回模型参数中 """
    # 确保 model_params 是可重复遍历的列表；解除主参数的扁平化
    model_params = list(model_params)
    temp = unflatten_master_params(model_params, master_params)
    
    # 原地写回模型中实际运行的 FP16 的 param
    for param, master_param in zip(model_params, temp):
        param.detach().copy_(master_param)


def unflatten_master_params(model_params, master_params):
    """ 解除主参数的扁平化，拆分成类似 model_params 的多个小 tensor """
    return _unflatten_dense_tensors(
        master_params[0].detach(), tuple(tensor for tensor in model_params)
    )

def zero_grad(model_params):
    """ 手动实现梯度清零，前向传播、反向传播所使用 """
    for param in model_params:
        if param.grad is not None:
            param.grad.detach_()  # 从当前计算图中脱离出来
            param.grad.zero_()  # 将梯度清零
