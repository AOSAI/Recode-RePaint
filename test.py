"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
与 image_sample.py类似, 但使用噪声图像分类器来引导采样过程, 以获得更真实的图像。
"""

import torch as th
import torch.nn.functional as F

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from models.diffusion import SamplerDDPM, SamplerDDIM
from tools import conf_base
from tools.script_util import (
    NUM_CLASSES, create_model_and_diffusion,
    create_classifier_model, load_config,
)

# 调整数据格式
def toU8(sample):
    if sample is None:
        return sample

    # 域值缩放：从 [-1, 1] 还原至 [0, 255]，并标准化为 uint8 格式
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    # 维度排列：从 [N, C, H, W] 转化为 [N, H, W, C]，保存图像的通用 shape
    sample = sample.permute(0, 2, 3, 1)
    # 内存连续化：有些操作后 Tensor 会变成非连续内存，可能导致 .numpy() 报错
    sample = sample.contiguous()
    # 去掉计算图、搬到 CPU 上、转化为 np 数组
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_base.Default_Conf):
    # ------------ 1. 配置加载与模型初始化 ------------
    print("Start", conf.sampling.name)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model, diffusion = create_model_and_diffusion(conf.model, conf.diffusion)
    model.load_state_dict(th.load(conf.sampling.model_path, map_location=device))
    model.to(device)
    if conf.model.use_fp16:
        model.convert_to_fp16()
    model.eval()
    show_progress = conf.sampling.show_progress
    
    # ------------ 2. 条件扩散模型的判断与初始化 ------------
    classifier_scale = conf.sampling.classifier_scale
    if classifier_scale > 0 and conf.sampling.classifier_path:
        classifier = create_classifier_model(**conf.classifier)
        classifier.load_state_dict(
            th.load(conf.sampling.classifier_path, map_location=device)
        )

        classifier.to(device)
        if conf.classifier.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * classifier_scale
    else:
        cond_fn = None

    # ------------ 3. 采样前的 model 更新、图像数据处理 ------------
    # 条件输入 y 必须存在，但通过 class_cond 决定是否真的使用
    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.model.class_cond else None, gt=gt)

    all_images = []  # 采样图像的保存列表
    dset = 'eval'
    eval_name = conf.get_default_eval_name()  # 获取自定义的 key1
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)  # 通过 key1 字典加载图像

    # ------------ 4. 循环采样 ------------
    print("sampling...")
    for batch in iter(dl):
        # 4.1 获取字典中的 key，原始图像和遮罩图像两个是 tensor
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        # 4.2 使用 model_kwargs 保存 gt 图像、mask 图像；计算 batch_size
        model_kwargs = {}
        model_kwargs["gt"] = batch['GT']
        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask
        batch_size = model_kwargs["gt"].shape[0]

        # 4.3 如果设置了 cond_y，使用指定的类别值；否则就随机选择类别
        # cond_y 在任何地方都没有被定义，但是 NoneDict 类中不会报错，直接使用第二个。
        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=device)
            model_kwargs["y"] = classes

        # 4.4 采样方式的选择：ddpm采样，ddim采样
        if not conf.model.use_ddim:
            sample_fn = SamplerDDPM(diffusion).sample
        else:
            sample_fn = SamplerDDIM(diffusion).sample
        
        # 4.5 执行采样
        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )

        # 4.6 还原采样过程中的四种图像：、原始 gt、被遮挡的输入 lrs、
        srs = toU8(result['sample'])  # 模型生成的 srs
        gts = toU8(result['gt'])  # 原始图像 gt
        # 被遮挡的输入图像：遮罩部分被填充为纯黑色，其余部分保留
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))
        gtkm = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))  # 遮罩图像 gtkm

        # 4.7 执行保存操作
        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gtkm,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False
        )

    print("sampling complete")


if __name__ == "__main__":
    # 自定义的处理 yaml 参数的对象
    conf_arg = conf_base.Default_Conf()
    # update 继承自 dict，用于将读取到的字典更新至 conf_arg 中
    conf_arg.update(load_config("./configs/template.yaml"))
    main(conf_arg)
