from piq import niqe
import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from imquality import brisque


# BRISQUE 局部缺陷修复的评价指标，可感知局部结构异常
def eval_by_brisque(folder):
    scores = []
    for fname in tqdm(sorted(os.listdir(folder))):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(folder, fname)
        img = Image.open(path).convert('RGB')
        score = brisque.score(img)
        scores.append(score)
    print(f'BRISQUE 平均分: {sum(scores)/len(scores):.4f}')


# NIQE 全图模糊修复，更适合图像锐化类任务
def eval_by_niqe(folder_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 可改
        transforms.ToTensor()
    ])
    scores = []
    for fname in tqdm(sorted(os.listdir(folder_path))):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(folder_path, fname)
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
        score = niqe(img_tensor, data_range=1.0).item()
        scores.append(score)
    print(f"NIQE 平均分: {sum(scores)/len(scores):.4f}")

if __name__ == "__main__":
    img_path = ""
    eval_by_niqe(img_path)
    eval_by_brisque(img_path)