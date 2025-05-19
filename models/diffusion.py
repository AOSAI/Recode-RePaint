import torch
import numpy as np
from collections import defaultdict
from models.noise_schedule import noise_related_calculate
from tools.scheduler import get_schedule_jump
from models.diff_utils import (
    ModelMeanType, ModelVarType, LossType, mean_flat, normal_kl, 
    extract, discretized_gaussian_log_likelihood
)


# -------------- 基类 --------------
class GaussianDiffusion:
    def __init__(
        self, *, betas, model_mean_type, model_var_type, loss_type, 
        rescale_timesteps=False
    ):
        # 将传入的参数注册为 self 全局变量
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # 获取 respace 后的 betas 序列，计算新的 num_timesteps
        self.betas = np.asarray(betas)
        self.num_timesteps = int(betas.shape[0])

        # 通过 betas 计算所有噪声调度参数，并注册为成员变量
        noise_schedule = noise_related_calculate(betas)
        for k, v in noise_schedule.items():
            setattr(self, k, v)

    def q_sample(self, x_start, t, noise=None):
        """从原始图像添加噪声, 模拟前向扩散过程的采样"""
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, t):
        """计算真实后验的均值和方差, 用于采样下一个时间步骤 x_(t-1)"""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def q_mean_variance(self, x_start, t):
        """向前加噪分布的直接表达式, 辅助函数, 在VLB计算中使用"""
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def predict_x0_from_eps(self, x_t, t, noise):
        """根据模型预测的噪声 ε，反推出原始图像 x_0 的估计值"""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_x0_from_xprev(self, x_t, t, xprev):
        """根据 x_t 和 x_(t-1)，反推 x_0 的估计值"""
        return (  # (xprev - coef2 * x_t) / coef1
            extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev - 
            extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape)
            * x_t
        )
    
    def predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """ DDIM 中使用，根据加噪公式，反推噪声"""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    
    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """返回模型预测的噪声均值、方差（固定、可学习）"""
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2] 
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # --------------- 可学习方差，通道数加倍 ---------------
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
                max_log = extract(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        elif self.model_var_type in [ModelVarType.FIXED_LARGE, ModelVarType.FIXED_SMALL]:
            # --------------- 固定方差 ---------------
            if self.model_var_type == ModelVarType.FIXED_LARGE:
                model_variance = np.append(self.posterior_variance[1], self.betas[1:])
                model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
            else:
                model_variance = self.posterior_variance
                model_log_variance = self.posterior_log_variance_clipped

            model_variance = extract(model_variance, t, x.shape)
            model_log_variance = extract(model_log_variance, t, x.shape)
        else:
            raise NotImplementedError(f"Unknown model_var_type: {self.model_var_type}")

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)  # 可选的去噪函数（如 super-resolution 后处理）
            if clip_denoised:
                return x.clamp(-1, 1)  # 保证像素值不越界
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            # --------------- 预测 x_t ---------------
            pred_xstart = process_xstart(
                self.predict_x0_from_xprev(x, t, model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            # --------------- 预测 x_0 或 epsilon ---------------
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self.predict_x0_from_eps(x, t, model_output)
                )
            model_mean, _, _ = self.q_posterior(pred_xstart, x, t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """ 估算变分下界 (Variational Lower Bound, VLB) """
        # --------------- 获取 真实后验分布 / 模型预测分布 的均值和方差 ---------------
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start, x_t, t)
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        # --------------- 计算 KL 散度；转换单位 nat 变 bit ---------------
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        # --------------- 计算负对数似然；转换单位 nat 变 bit ---------------
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # --------------- 在第一个时间步返回解码器 NLL，否则返回 KL ---------------
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """损失函数调用入口, 用于计算单个时间步下的损失值"""
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)  # 加噪音得到 x_t

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model, x_start=x_start, x_t=x_t, t=t,
                clip_denoised=False, model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                # loss 尺度跟 MSE 类型对齐
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, t, **model_kwargs)
            
            # ------------ 如果方差可学习，使用 KL/NLL 计算 ------------
            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # 利用变异约束学习方差，但不要让它影响我们对平均值的预测。
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start, x_t=x_t, t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # 缩放修正项，保持 vb_loss 和 mes_loss 的数值一致性
                    terms["vb"] *= self.num_timesteps / 1000.0

            # ------------ 使用 MSE 计算均值 ------------
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)

            # ------------ 如果可学习方差 vb 存在，合并 MSE ------------
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms
    
    
# -------------- DDPM 原始采样器 --------------
class SamplerDDPM:
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion

        # 用到的属性，重新建立索引引用，不会额外占用显存
        self.p_mean_variance = diffusion.p_mean_variance
        self.num_timesteps = diffusion.num_timesteps
        self.alphas_cumprod = diffusion.alphas_cumprod
        self.betas = diffusion.betas

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        给定函数 cond_fn, 计算条件对数概率相对于 x 的梯度，计算上一步的均值。
        具体而言, cond_fn 计算 grad(log(p(y|x)))，我们希望以 y 为条件。
        这采用了 Sohl-Dickstein 等人 (2015 年) 的条件策略。
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean
    
    def undo(self, img_out, t):
        """ RePaint 回跳时的加噪过程 """
        beta = extract(self.betas, t, img_out.shape)
        noise = torch.randn_like(img_out)
        return torch.sqrt(1 - beta) * img_out + torch.sqrt(beta) * noise

    def p_sample(
        self, model, x_t, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, conf=None, pred_xstart=None,
    ):
        """ Sample x_{t-1} from the model at the given timestep. """

        # 如果启用了 "前一步注入引导策略"
        if conf.inpa_inj_sched_prev:
            # 只有当前时刻，已经有预测的 x_start 时，才进行注入操作
            if pred_xstart is not None:
                # 获取 二值 mask，以及原始输入图像
                gt_keep_mask = model_kwargs.get('gt_keep_mask')
                gt = model_kwargs['gt']

                # 通过前向加噪公式，手动计算 weighed_gt
                alpha_cumprod = extract(self.alphas_cumprod, t, x_t.shape)
                gt_weight = torch.sqrt(alpha_cumprod)
                gt_part = gt_weight * gt
                noise_weight = torch.sqrt((1 - alpha_cumprod))
                noise_part = noise_weight * torch.randn_like(x_t)
                weighed_gt = gt_part + noise_part

                # gt_keep_mask 为 1 的地方使用 weighed_gt；为 0 的地方使用原图
                x_t = gt_keep_mask * ( weighed_gt) + (1 - gt_keep_mask) * (x_t)

        out = self.p_mean_variance(
            model, x_t, t, model_kwargs=model_kwargs,
            clip_denoised=clip_denoised, denoised_fn=denoised_fn,
        )

        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x_t, t, model_kwargs=model_kwargs
            )

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        
        return {
            "sample": sample, 
            "pred_xstart": out["pred_xstart"],
            'gt': model_kwargs.get('gt'),
        }
    
    def p_sample_loop(
        self, model, shape, device=None, noise=None, progress=False, cond_fn=None,
        clip_denoised=True, denoised_fn=None, model_kwargs=None, conf=None
    ):
        """ 从纯噪声开始反复调用 p_sample 采样出最终图像 """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = torch.randn(*shape, device=device)

        pred_xstart = None
        sample_idxs = defaultdict(lambda: 0)

        if conf.schedule_jump_params:
            # 获取 RePaint 回跳机制下的 新时间步
            times = get_schedule_jump(**conf.schedule_jump_params)

            # 构造 “相邻时间对”；是否启用进度条显示
            time_pairs = list(zip(times[:-1], times[1:]))
            if progress:
                from tqdm.auto import tqdm
                time_pairs = tqdm(time_pairs)

            for t_last, t_cur in time_pairs:
                t_last_t = torch.tensor([t_last] * shape[0], device=device)
                
                # 如果当前 t 在减小，逆向生成；否则进行回跳加噪
                if t_cur < t_last:
                    with torch.no_grad():
                        image_before_step = image_after_step.clone()
                        out = self.p_sample(
                            model, image_after_step, t_last_t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            cond_fn=cond_fn,
                            model_kwargs=model_kwargs,
                            conf=conf,
                            pred_xstart=pred_xstart
                        )
                        image_after_step = out["sample"]
                        pred_xstart = out["pred_xstart"]
                        sample_idxs[t_cur] += 1
                        yield out
                else:
                    t_shift = conf.get('inpa_inj_time_shift', 1)  # 没有该键则使用 1
                    # image_before_step = image_after_step.clone()  # 保存当前状态
                    
                    # 根据 DDPM 的公式重新加噪
                    image_after_step = self.undo(image_after_step, t=t_last_t+t_shift)

                    # 保存当前步预测出来的 x_0，用于下一步的融合/参考
                    pred_xstart = out["pred_xstart"]

    
    def sample(
        self, model, batch_size, image_size, device, clip_denoised=True, model_kwargs=None,
        cond_fn=None, progress=True, return_all=False, conf=None
    ):
        """外部调用入口，封装 p_sample_loop"""
        shape = (batch_size, 3, image_size, image_size)
        final = None
        for sample in self.p_sample_loop(
            model, shape, device, 
            clip_denoised=clip_denoised, 
            model_kwargs=model_kwargs,
            cond_fn=cond_fn, 
            progress=progress, 
            conf=conf
        ):
            final = sample
        return final if return_all else final["sample"]
    

# -------------- DDIM 加速采样器 --------------
class SamplerDDIM:
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion
        
        # -------------- ddim_sample、ddim_reverse_sample --------------
        self.p_mean_variance = diffusion.p_mean_variance
        self.predict_eps_from_xstart = diffusion.predict_eps_from_xstart
        self.alphas_cumprod = diffusion.alphas_cumprod
        self.alphas_cumprod_prev = diffusion.alphas_cumprod_prev
        self.alphas_cumprod_next = diffusion.alphas_cumprod_next
        self.num_timesteps = diffusion.num_timesteps

    def ddim_reverse_sample(
        self, model, x, t, eta=0.0,
        clip_denoised=True, denoised_fn=None, model_kwargs=None,
    ):
        """ 用于 反推加噪 轨迹，计算 x_(t+1) """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        )

        # 在使用 x_start 或 x_prev 预测的情况下，重新得出ε
        eps = self.predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample(
        self, model, x, t, eta=0.0,
        clip_denoised=True, denoised_fn=None, model_kwargs=None,
    ):
        """ Sample x_{t-1} from the model using DDIM. """
        # -------------- 返回 mean、variance、log_variance、pred_xstart 字典 --------------
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        )
        # -------------- 重新计算噪声分布的方差，使用 eta 控制随机程度 --------------
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # -------------- 在使用 x_start 或 x_prev 预测的情况下，重新得出ε --------------
        eps = self.predict_eps_from_xstart(x, t, out["pred_xstart"])
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        noise = torch.randn_like(x)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def sample_loop(
        self, model, shape, device=None, noise=None, progress=False,
        clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0,
    ):
        """ 使用 DDIM 对模型进行采样，并从 DDIM 的每个时间步产生中间样本 """
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        
        indices = reversed(range(self.num_timesteps))
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        with torch.no_grad():
            for i in indices:
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                img = self.ddim_sample(
                    model, img, t, eta=eta, model_kwargs=model_kwargs,
                    clip_denoised=clip_denoised, denoised_fn=denoised_fn
                )

        return img["sample"]
    
    def sample(
        self, model, image_size, batch_size=16, clip_denoised=True, model_kwargs=None
    ):
        """外部调用入口，封装 sample_loop"""
        shape = (batch_size, 3, image_size, image_size)
        device = next(model.parameters()).device
        return self.sample_loop(
            model, shape, device, 
            clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
    
    
