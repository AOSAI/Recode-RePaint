import torch
import os
from tqdm import tqdm
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image
import cv2

def generate_ev2col_mask(height, width):
    """
    每两列遮一列的竖条纹遮挡 (ev2li)
    返回值: torch.Tensor of shape [1, H, W], 值为 0/1
    """
    mask = torch.ones((height, width), dtype=torch.uint8)
    mask[:, ::2] = 0  # 每两列遮一列（0 = 被遮挡）
    return mask.unsqueeze(0)  # 添加通道维度 → [1, H, W]

def generate_ev2row_mask(height, width):
    """
    每两行遮一行的横条纹遮挡 (ev2row)
    返回值: torch.Tensor of shape [1, H, W], 值为 0/1
    """
    mask = torch.ones((height, width), dtype=torch.uint8)
    mask[::2, :] = 0  # 每两行遮一行
    return mask.unsqueeze(0)

def generate_ev4row_mask(height, width):
    """ 每四行遮一行的横条纹遮挡 (ev4row) """
    mask = torch.ones((height, width), dtype=torch.uint8)
    mask[::4, :] = 0  # 每两行遮一行
    return mask.unsqueeze(0)

def generate_ev4col_mask(height, width):
    """ 每四行遮一行的横条纹遮挡 (ev4row) """
    mask = torch.ones((height, width), dtype=torch.uint8)
    mask[:, ::4] = 0  # 每两行遮一行
    return mask.unsqueeze(0)

def save_mask1(mask_tensor, save_path):
    """
    将单通道遮挡 mask（[1, H, W]）保存为图像
    """
    mask_img = mask_tensor.squeeze(0) * 255  # 0/1 → 0/255
    mask_img = TF.to_pil_image(mask_img)
    mask_img.save(save_path)
    # print(f"✅ 保存成功：{save_path}")


def method_choose1(method_name, out_dir, H, W, filename):
    if method_name == "ev2li":
        mask1 = generate_ev2col_mask(H, W)
        save_mask1(mask1, os.path.join(out_dir, f"{filename}.png"))
    elif method_name == "ev2row":
        mask2 = generate_ev2row_mask(H, W)
        save_mask1(mask2, os.path.join(out_dir, f"{filename}.png"))
    elif method_name == "ev4row":
        mask2 = generate_ev4row_mask(H, W)
        save_mask1(mask2, os.path.join(out_dir, f"{filename}.png"))
    elif method_name == "ev4col":
        mask2 = generate_ev4col_mask(H, W)
        save_mask1(mask2, os.path.join(out_dir, f"{filename}.png"))
    else:
        raise ValueError("don't have this method")

# --------------- Generate Random Line Masks ---------------
def save_mask2(mask_img, save_path):
    if mask_img.dtype != np.uint8:
        mask_img = mask_img.astype(np.uint8)

    max_val = mask_img.max()
    if max_val == 1:
        mask_img = mask_img * 255

    img = Image.fromarray(mask_img)
    img.save(save_path)

def gen_postion_list(length, num_mask, gap_min):
    postion_list = np.random.randint(0, length - 1, size=num_mask)
    postion_list.sort()

    if postion_list[-1] - postion_list[0] > (length - gap_min):
        return gen_postion_list(length, num_mask, gap_min)
    else:
        return postion_list
    
def random_line_mask(
    length: int,
    min_single: float = 0.02,
    max_single: float = 0.1,
    max_total: float = 0.3
) -> np.ndarray:
    
    mask_list = []
    while True:
        if max_total - sum(mask_list) < max_single:
            mask_single = round(max_total - sum(mask_list), 2)
            mask_list.append(mask_single)
            break

        mask_single = round(random.uniform(min_single, max_single), 2)
        mask_list.append(mask_single)

    gap_min = int(length * (min_single + max_single))
    num_mask = len(mask_list)
    postion_list = gen_postion_list(length, num_mask, gap_min)
    
    center_postion = int(num_mask/2 - 1) if (num_mask%2 == 0) else int(num_mask/2)
    left_index = center_postion
    right_index = center_postion

    while True:
        prev_index = left_index - 1
        next_index = right_index + 1

        if next_index == num_mask:
            last_mask = int(length * mask_list[-1])
            re_dis = postion_list[-1] - (length - last_mask - 1)
            
            if postion_list[0] < 0:
                first_postion = postion_list[0]
                postion_list = [x - first_postion for x in postion_list]
            if re_dis > 0:
                postion_list = [x - re_dis for x in postion_list]
            
            break

        if prev_index >= 0:
            distance = postion_list[left_index] - postion_list[prev_index]
            if distance < gap_min:
                postion_list[prev_index] = postion_list[left_index] - gap_min
            left_index = prev_index
        
        if next_index < num_mask:
            distance = postion_list[right_index] - postion_list[next_index]
            if distance < gap_min:
                postion_list[next_index] = postion_list[right_index] + gap_min
            right_index = next_index

    return mask_list, postion_list

def gen_rali_mask(mask_list, postion_list, mask_img, length, flag):
    for ratio, postion in zip(mask_list, postion_list):
        start = int(postion)
        start = max(start, 0)  # 防止越界
        end = start + int(round(length * ratio))
        end = min(end, length - 1)  # 防止越界

        if flag == "row":
            mask_img[start:end, :] = 0
        if flag == "col":
            mask_img[:, start:end] = 0
    
    return mask_img

def row_col_process(
    height: int,
    width: int,
    min_single: float = 0.02,
    max_single: float = 0.1,
    max_total: float = 0.3,
    flag: str = "row",
    out_dir: str = None,      
    filename: str = None,      
) -> np.ndarray:
    
    if flag == "row":
        mask_list, postion_list = random_line_mask(height, min_single, max_single, max_total)
        mask_img = np.ones((height, width), dtype=np.uint8)
        mask_img = gen_rali_mask(mask_list, postion_list, mask_img, height, flag)

    if flag == "col":
        mask_list, postion_list = random_line_mask(width, min_single, max_single, max_total)
        mask_img = np.ones((height, width), dtype=np.uint8)
        mask_img = gen_rali_mask(mask_list, postion_list, mask_img, width, flag)
    
    if flag == "both":
        mask_list, postion_list = random_line_mask(height, min_single, max_single, max_total)
        mask_img = np.ones((height, width), dtype=np.uint8)
        mask_img = gen_rali_mask(mask_list, postion_list, mask_img, height, flag="row")
        
        mask_list, postion_list = random_line_mask(width, min_single, max_single, max_total)
        mask_img = gen_rali_mask(mask_list, postion_list, mask_img, width, flag="col")

    # return mask_img
    save_mask2(mask_img, os.path.join(out_dir, f"{filename}.png"))


def generate_random_circle_mask(
    H, W, out_dir, filename,
    mask_value=0,
    max_mask_ratio=0.25,
    min_radius=10,
    max_radius=50,
    seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    mask = np.ones((H, W), dtype=np.uint8) * 255  # 初始化为全白
    total_area = H * W
    masked_area = 0
    max_mask_area = total_area * max_mask_ratio

    attempt = 0
    max_attempts = 1000  # 防止死循环

    while masked_area < max_mask_area and attempt < max_attempts:
        r = random.randint(min_radius, max_radius)
        cx = random.randint(r, W - r - 1)
        cy = random.randint(r, H - r - 1)

        temp_mask = np.zeros_like(mask)
        cv2.circle(temp_mask, (cx, cy), r, 1, thickness=-1)  # 圆内部为1

        new_mask_area = np.sum((temp_mask == 1) & (mask == 255))
        if masked_area + new_mask_area > max_mask_area:
            attempt += 1
            continue

        mask[temp_mask == 1] = mask_value
        masked_area += new_mask_area
        attempt += 1

    # return mask
    save_mask2(mask, os.path.join(out_dir, f"{filename}.png"))


if __name__ == "__main__":

    H, W = 256, 256
    mask_list = ["ev2col", "ev2row", "ev4col", "ev4row"]
    flag_list = ["row", "col", "both"]

    img_dir= "D:\CodeReproduction\Recode-RePaint\yolo\images_and_masks\\img_circle"
    out_dir = "D:\CodeReproduction\Recode-RePaint\yolo\images_and_masks\mask_randcircle"
    os.makedirs(out_dir, exist_ok=True)

    exts = ['.jpg', '.jpeg', '.png']
    files = os.listdir(img_dir)
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in exts]

    for idx, filename in enumerate(tqdm(image_files)):
        # img_path = os.path.join(img_dir, filename)
        # method_choose1(mask_list[2], out_dir, H, W, filename)
        filename1 = os.path.splitext(filename)[0]
        # row_col_process(H, W, 0.02, 0.08, 0.25, flag_list[1], out_dir, filename1)

        mask = generate_random_circle_mask(
            H, W, out_dir, filename1,
            mask_value=0,
            max_mask_ratio=0.20,
            min_radius=10,
            max_radius=30,
            seed=None,
        )



