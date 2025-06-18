import os
import numpy as np
import torch as th
from PIL import Image
from tools import logger
from models.diffusion import SamplerDDPM, SamplerDDIM
from tools.script_util import (
    NUM_CLASSES, create_model_and_diffusion, load_config,
)

def main(sample_type):
    # ------------ 参数字典、硬件设备、日志文件的初始化 ------------
    config = load_config("./configs/test_128.yml")
    args_s = config['sampling']
    args_m = config['model']
    args_d = config['diffusion']
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    logger.configure()

    # ------------ 扩散模型、神经网络的初始化 ------------
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args_m, args_d)
    model.load_state_dict(
        th.load(args_s["model_path"], map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # ------------ 循环采样 ------------
    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) < args_s['num_samples']:
        # 如果是类别条件生成，随机生成 class label
        model_kwargs = {}
        if args_m['class_cond']:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args_s["batch_size"],), device=device
            )
            model_kwargs["y"] = classes
        
        # 采样方式选择：ddpm采样，ddim采样
        if not args_s['use_ddim']:
            sample_fn = SamplerDDPM(diffusion).sample
        else:
            sample_fn = SamplerDDIM(diffusion).sample

        # 调用采样方法，生成一批图片
        sample = sample_fn(
            model,
            args_m['image_size'],
            batch_size = args_s['batch_size'], 
            clip_denoised = args_s['clip_denoised'],
            model_kwargs = model_kwargs,
        )

        # 调整数据格式（值域缩放、排列维度）
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # all_images、all_labels 中存入当前批次的 图像、标签
        all_images.extend(sample.cpu().numpy())
        if args_m['class_cond']:
            all_labels.extend(classes.cpu().numpy())
        logger.log(f"created {len(all_images)} samples")

    # ------------ 保存类型：图像、numpy数组 ------------
    if "image" in sample_type:
        save_image(args_s, args_m, all_images, all_labels)
    if "nparray" in sample_type:
        save_nparray(args_s, args_m, all_images, all_labels)


def save_image(args_s, args_m, all_images, all_labels):
    save_dir = os.path.join(logger.get_dir(), "sampled_images")
    os.makedirs(save_dir, exist_ok=True)

    for i, img_arr in enumerate(all_images[: args_s["num_samples"]]):
        img = Image.fromarray(img_arr)
        if args_m["class_cond"]:
            label = all_labels[i]
            filename = f"{label}_{i:06d}.png"
        else:
            filename = f"{i:06d}.png"
        img.save(os.path.join(save_dir, filename))

    logger.log(f"saved {args_s['num_samples']} images to {save_dir}")
    

def save_nparray(args_s, args_m, all_images, all_labels):
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args_s["num_samples"]]
    
    if args_m['class_cond']:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args_s["num_samples"]]
    
    # 假设有 1000 次采样，shape_str = “1000x64x64x3”
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    
    # 类别条件生成保存两个数组，无条件生成只保存图像数组
    if args_m['class_cond']:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    logger.log("sampling complete")


if __name__ == "__main__":
    """
    采样生成图像: image
    采样生成 numpy 数组: nparray (用于 FID 评估)
    """
    main(["image", "nparray"])
