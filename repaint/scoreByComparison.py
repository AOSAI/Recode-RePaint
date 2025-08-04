"""
FID(整体图像的分布相似): [<10] 几乎与GT一致, [<30] 感知质量良好, [<50] 中等偏差,肉眼可见, [>50] 质量较差
LPIPS(每张图像的感知差异): [<0.1] 几乎无差异, [<0.3] 相似度高, [<0.5] 感知差异较明显, [>0.5] 相差较大
PSNR(像素误差MSE): [>30dB] 高质量修复, [>20] 可接受的人眼效果, [<20] 噪声过大/失真过多
SSIM(结构相似性): [>0.95] 几乎完美还原, [>0.8] 良好结构合理, [>0.6] 有些结构偏差, [<0.6] 结构明显失真
"""
import os
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from tqdm import tqdm
import lpips
from torchvision.models import alexnet, AlexNet_Weights
import torchvision.models as models
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

# 使用 SSIM 评估图像
def eval_by_ssim(real_images, fake_images, data_range=255.0):
    """
    real_images, fake_images: List of NumPy arrays (H, W, C) or (H, W)
    data_range: 最大像素值（255.0 for uint8, 1.0 for float）
    """
    assert len(real_images) == len(fake_images), "Image list lengths must match"

    scores = []
    for real, fake in zip(real_images, fake_images):
        # 如果是彩色图像，需要指定通道轴
        if real.ndim == 3:
            ssim_val = ssim(real, fake, data_range=data_range, channel_axis=2)
        else:
            ssim_val = ssim(real, fake, data_range=data_range)

        scores.append(ssim_val)

    avg_score = sum(scores) / len(scores)
    print(f'SSIM score: {avg_score:.4f}')

# 使用 PSNR 评估图像
def eval_by_psnr(real_images, fake_images, data_range=255.0):
    """
    real_images, fake_images: List of NumPy arrays (H, W, C) or (H, W)
    data_range: 255.0 for uint8 images, 1.0 for normalized floats
    """
    assert len(real_images) == len(fake_images), "Mismatched number of images"
    
    scores = []
    for real, fake in zip(real_images, fake_images):
        psnr_val = peak_signal_noise_ratio(real, fake, data_range=data_range)
        scores.append(psnr_val)  # no .item() needed
    
    avg_score = sum(scores) / len(scores)
    print(f'PSNR score: {avg_score:.4f} dB')

# 使用 LPIPS 评估图像
def eval_by_lpips(real_images, fake_images, loss_fn, net='alex'):
    assert real_images.shape == fake_images.shape, "LPIPS requires same shape for real and fake images"

    scores = []
    for real, fake in zip(real_images, fake_images):
        d = loss_fn(real.unsqueeze(0), fake.unsqueeze(0))  # batch size=1
        scores.append(d.item())

    avg_score = sum(scores) / len(scores)
    print(f'LPIPS ({net}) score: {avg_score:.4f}')


# 使用 FID 评估图像（只能 GPU，默认 float64）
def eval_by_fid_float64(real_images, fake_images, device):
    # 创建 FID 评估器, 2048 是 InceptionV3 的最后特征层输出
    fid = FrechetInceptionDistance(feature=2048).to(device).float()

    # 添加图像集合
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    # 计算 FID
    score = fid.compute().item()
    print(f'FID score: {score:.4f}')


# 加载两个文件夹的图像
def load_images_from_folder(folder, preprocess):
    images = []
    filenames = sorted(os.listdir(folder))  # 建议排序，确保图像一一对应
    for filename in tqdm(filenames):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # 跳过非图像文件（比如 .DS_Store）

        path = os.path.join(folder, filename)
        image = Image.open(path).convert('RGB')
        image = preprocess(image)
        images.append(image)
    return torch.stack(images)


def load_images_for_pixel_metrics(folder):
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(folder, filename)
        image = Image.open(path).convert('RGB')
        images.append(np.array(image))  # 返回原始像素数据
    return images


if __name__ == "__main__":
    # GT 图像和 Repaired图像文件夹路径, 加载图像
    real_dir = './repaint/log/Comparison/0-gt-all'
    fake_dir = './repaint/log/Comparison/0-inpainted-places'
    # fake_dir = './repaint/log/Comparison/0-inpainted-mine'

    device = (
        # "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # # FID 预处理（[0,1]，不用归一化）
    preprocess_fid = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.PILToTensor()  # 这里不归一化，保持uint8格式
    ])

    real_images = load_images_from_folder(real_dir, preprocess_fid).to(device)
    fake_images = load_images_from_folder(fake_dir, preprocess_fid).to(device)
    # ========== FID ==========
    eval_by_fid_float64(real_images, fake_images, device)


    # LPIPS 的预处理（[-1,1]，需要归一化）
    preprocess_lpips = transforms.Compose([
        transforms.Resize((256, 256)),  # 只要 real 和 fake 一样就行
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 映射到 [-1, 1]
    ])
    real_lpips = load_images_from_folder(real_dir, preprocess_lpips).to(device)
    fake_lpips = load_images_from_folder(fake_dir, preprocess_lpips).to(device)
    # ========== LPIPS ==========
    models.alexnet = lambda **kwargs: alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    lpips_net="alex"  # 支持 net='alex'（默认）、'vgg'、'squeeze'
    loss_fn = lpips.LPIPS(net=lpips_net).to(device).float()  # 初始化感知损失函数
    eval_by_lpips(real_lpips, fake_lpips, loss_fn, net=lpips_net)

    
    real_imgs = load_images_for_pixel_metrics(real_dir)
    fake_imgs = load_images_for_pixel_metrics(fake_dir)
    # ========== PSNR ==========
    eval_by_psnr(real_imgs, fake_imgs)
    # ========== SSIM ==========
    eval_by_ssim(real_imgs, fake_imgs)






