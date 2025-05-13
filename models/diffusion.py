import torch
import numpy as np
from noise_schedule import noise_related_calculate
from diff_utils import (ModelMeanType, ModelVarType, LossType, 
                    extract, mean_flat, normal_kl, 
                    discretized_gaussian_log_likelihood)


# -------------- 基类 --------------
class GaussianDiffusion:
    def __init__(self, *, betas, model_mean_type, model_var_type, 
                 loss_type, rescale_timesteps=False):
        # 将传入的参数注册为 self 全局变量
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        self.betas = np.asarray(betas)
        self.num_timesteps = int(betas.shape[0])

        # 获取所有噪声调度参数，并注册为 self.xxx
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
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # --------------- 可学习方差，通道数加倍 ---------------
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
                max_log = extract(torch.log(self.betas), t, x.shape)
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
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

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
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            
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
    
    def _prior_bpd(self, x_start):
        """
        获取变分下限的先验 KL 项，单位为比特/比特。该项无法优化，因为它只取决于编码器。

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        计算整个变分下限（以每比特为单位）以及其他相关数量。

        :param model: 评估损失的模型
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: 每个批次元素的总变分下限。
                 - prior_bpd: 下限中的前项。
                 - vb: 下界项的 [N x T] 张量。
                 - xstart_mse: 每个时间步的 x_0 MSE 的 [N x T] 张量。
                 - mse: 每个时间步的ε MSE 的 [N x T] 张量。
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }
    
    
# -------------- DDPM 原始采样器 --------------
class SamplerDDPM:
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion

        # 用到的属性，重新建立索引引用，不会额外占用显存
        self.p_mean_variance = diffusion.p_mean_variance
        self.num_timesteps = diffusion.num_timesteps

    def p_sample(
        self, model, x_t, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """ Sample x_{t-1} from the model at the given timestep. """
        out = self.p_mean_variance(
            model, x_t, t, model_kwargs=model_kwargs,
            clip_denoised=clip_denoised, denoised_fn=denoised_fn,
        )

        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def p_sample_loop(
        self, model, shape, device=None, noise=None, progress=False,
        clip_denoised=True, denoised_fn=None, model_kwargs=None,
    ):
        """ 从纯噪声开始反复调用 p_sample 采样出最终图像 """
        if noise is not None:
            x_t = noise
        else:
            x_t = torch.randn(shape, device=device)
        
        indices = reversed(range(self.num_timesteps))
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        with torch.no_grad():
            for i in indices:
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                x_t = self.p_sample(
                    model, x_t, t, model_kwargs=model_kwargs,
                    clip_denoised=clip_denoised, denoised_fn=denoised_fn
                )

        return x_t["sample"]
    
    def sample(
        self, model, image_size, batch_size=16, clip_denoised=True, model_kwargs=None
    ):
        """外部调用入口，封装 p_sample_loop"""
        shape = (batch_size, 3, image_size, image_size)
        device = next(model.parameters()).device  # 👈 自动获取模型所在设备
        return self.p_sample_loop(
            model, shape, device, 
            clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
    
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
    
    
