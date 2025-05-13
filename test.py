"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
与 image_sample.py类似, 但使用噪声图像分类器来引导采样过程, 以获得更真实的图像。
"""

import os
import torch as th
import torch.nn.functional as F
import time
import argparse

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from tools import conf_base
from tools.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
    load_config,
)  # noqa: E402

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
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
        classifier = create_classifier(**conf.classifier)
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

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if args_m["class_cond"] else None, gt=gt)

    print("sampling...")
    all_images = []

    dset = 'eval'
    eval_name = conf.get_default_eval_name()
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in iter(dl):
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}
        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )


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
        srs = toU8(result['sample'])
        gts = toU8(result['gt'])
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")


if __name__ == "__main__":
    # 自定义的处理 yaml 参数的对象
    conf_arg = conf_base.Default_Conf()
    # update 继承自 dict，用于将读取到的字典更新至 conf_arg 中
    conf_arg.update(load_config("./configs/template.yaml"))
    main(conf_arg)
