import torch
from collections import defaultdict
from tools.scheduler import get_schedule_jump, get_schedule_jump2
from models.diffusion import GaussianDiffusion
from models.diff_utils import extract


# -------------- DDPM 原始采样器 RePaint 版 --------------
class SamplerRePaint1:
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion

        # 用到的属性，重新建立索引引用，不会额外占用显存
        self.p_mean_variance = diffusion.p_mean_variance
        self.num_timesteps = diffusion.num_timesteps
        self.alphas_cumprod = diffusion.alphas_cumprod
        self.betas = diffusion.betas
        self.q_sample = diffusion.q_sample

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
        """ RePaint 回跳时的加噪过程 1, 每一步都引入独立的高斯噪声"""
        beta = extract(self.betas, t, img_out.shape)
        noise = torch.randn_like(img_out)
        return torch.sqrt(1 - beta) * img_out + torch.sqrt(beta) * noise

    def p_sample(
        self, model, x_t, t, clip_denoised=True, denoised_fn=None, cond_fn=None, 
        model_kwargs=None, conf=None, pred_xstart=None,
    ):
        """ Sample x_{t-1} from the model at the given timestep. """

        # 如果启用了 "前一步注入引导策略"
        if conf.sampling.process_xt:
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
                    if conf.sampling.add_noise_once:
                        t_shift = int(conf.schedule_jump_params.jump_length) - 1
                        assert t_cur - t_last == t_shift, "jump steps have mistakes"
                        image_after_step = self.q_sample(image_after_step, t=t_last_t+t_shift)
                    else:
                        t_shift = 1
                        image_after_step = self.undo(image_after_step, t=t_last_t+t_shift)

    
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
    

# -------------- DDPM 原始采样器 RePaint+ 版 --------------
class SamplerRePaint2:
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion

        # 用到的属性，重新建立索引引用，不会额外占用显存
        self.p_mean_variance = diffusion.p_mean_variance
        self.num_timesteps = diffusion.num_timesteps
        self.alphas_cumprod = diffusion.alphas_cumprod
        self.betas = diffusion.betas
        self.q_sample = diffusion.q_sample

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
    
    def undo(self, pred_x0, t, noise=None):
        """ RePaint+ 回跳时的加噪, 直接使用预测的 x_start, 而不是 x_t """
        beta = extract(self.betas, t, pred_x0.shape)
        if noise is None:
            noise = torch.randn_like(pred_x0)
        return torch.sqrt(1 - beta) * pred_x0 + torch.sqrt(beta) * noise

    def p_sample(
        self, model, x_t, t, clip_denoised=True, denoised_fn=None, cond_fn=None, 
        model_kwargs=None, conf=None, pred_xstart=None
    ):
        """ Sample x_{t-1} from the model at the given timestep. """

        if conf.sampling.process_xt:
            if pred_xstart is not None:
                gt_keep_mask = model_kwargs.get('gt_keep_mask')  # 获取 二值 mask
                gt = model_kwargs['gt']  # 获取输入的原始图像

                # 通过前向加噪公式，手动计算 weighed_gt
                alpha_cumprod = extract(self.alphas_cumprod, t, x_t.shape)
                gt_weight = torch.sqrt(alpha_cumprod)
                gt_part = gt_weight * gt
                noise_weight = torch.sqrt((1 - alpha_cumprod))
                noise_part = noise_weight * torch.randn_like(x_t)
                weighed_gt = gt_part + noise_part

                # gt_keep_mask 为 1 的地方使用 weighed_gt；为 0 的地方使用原图
                x_t = gt_keep_mask * (weighed_gt) + (1 - gt_keep_mask) * (x_t)

        out = self.p_mean_variance(
            model, x_t, t, model_kwargs=model_kwargs,
            clip_denoised=clip_denoised, denoised_fn=denoised_fn,
        )

        if conf.sampling.process_xstart:
            if pred_xstart is not None:
                gt = model_kwargs["gt"]
                gt_keep_mask = model_kwargs.get("gt_keep_mask")
                out["pred_xstart"] = gt_keep_mask * gt + (1 - gt_keep_mask) * out["pred_xstart"]

        if conf.sampling.fix_seed:
            torch.manual_seed(1234)

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
        addnoise_base = None

        if conf.schedule_jump_params:
            # 获取 RePaint 回跳机制下的 新时间步
            times = get_schedule_jump2(**conf.schedule_jump_params2)

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
                    if conf.sampling.add_noise_once:
                        t_shift = int(conf.schedule_jump_params.jump_length) - 1
                        assert t_cur - t_last == t_shift, "jump steps have mistakes"
                        # 一次性加噪（跳步）
                        if conf.sampling.process_xstart:
                            addnoise_base = pred_xstart
                        else:
                            addnoise_base = image_after_step
                        image_after_step = self.q_sample(addnoise_base, t=t_last_t+t_shift)
                    else:
                        t_shift = 1
                        # 多次逐步加噪（每次 +1）
                        if conf.sampling.process_xstart:
                            if t_last + 1 == t_cur:
                                addnoise_base = pred_xstart
                            else:
                                addnoise_base = image_after_step
                        else:
                            addnoise_base = image_after_step
                        image_after_step = self.undo(addnoise_base, t=t_last_t+t_shift)
                    
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
    