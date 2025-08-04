import cv2
import os
from tqdm import tqdm
import numpy as np
from img_common import imread_unicode, imwrite_unicode
from PIL import Image

# ------------ 1. Image Resize Functions ------------
def resize_to_small(img, target_size, scale_factor=0.5, interpolation=cv2.INTER_AREA):
    # 1.1 确定目标宽高
    target_h, target_w = target_size
    h, w = img.shape[:2]

    # 1.2 如果当前尺寸等于目标尺寸，直接返回
    if (h, w) == (target_h, target_w):
        return img

    # 1.3 设定缩放系数，循环多次缩放，直到接近目标
    while max(h, w) * scale_factor > max(target_h, target_w) * 1.9:
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        h, w = img.shape[:2]

    # 1.4 最后一步精确缩放
    if (h, w) != (target_h, target_w):
        final_interp = interpolation
        if target_h > h or target_w > w:
            final_interp = cv2.INTER_CUBIC  # 放大时用 CUBIC，质量更好
        img = cv2.resize(img, (target_w, target_h), interpolation=final_interp)
    
    return img
    
def resize_to_large(img, target_size, scale_factor=2.0, interpolation=cv2.INTER_CUBIC):
    target_h, target_w = target_size
    h, w = img.shape[:2]

    while max(h, w) * scale_factor < max(target_h, target_w) / 1.9:
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        h, w = img.shape[:2]

    if (h, w) != (target_h, target_w):
        final_interp = interpolation
        if target_h < h or target_w < w:
            final_interp = cv2.INTER_AREA  # 缩小时用 AREA，质量更好
        img = cv2.resize(img, (target_w, target_h), interpolation=final_interp)

    return img


# ------------ 2. Image Cropping: center or silding windows by ratio ------------
def crop_by_ratio(img, ratio_thresh=1.2):
    h, w = img.shape[:2]
    crops = []

    long_edge = max(h, w)
    short_edge = min(h, w)
    ratio = long_edge / short_edge
    if ratio > 1.7:
        stride = int(short_edge * 0.5 + 1)
    else: 
        stride = int(short_edge * (ratio - 1) + 1)

    if ratio > ratio_thresh:
        # Crop by sliding window
        if h > w:
            # 竖直滑动
            for top in range(0, h - short_edge + 1, stride):
                crop = img[top:top + short_edge, 0:w]
                crops.append(crop)
            # 末尾补裁（防止滑不到最后一块）
            if h % stride != 0 and (h - short_edge) % stride != 0:
                crop = img[-short_edge:, 0:w]
                crops.append(crop)
        else:
            # 水平滑动
            for left in range(0, w - short_edge + 1, stride):
                crop = img[0:h, left:left + short_edge]
                crops.append(crop)
            # 末尾补裁
            if w % stride != 0 and (w - short_edge) % stride != 0:
                crop = img[0:h, -short_edge:]
                crops.append(crop)
    else:
        # Crop by center
        center_h = h // 2
        center_w = w // 2
        half = short_edge // 2
        crop = img[center_h - half:center_h + half, center_w - half:center_w + half]
        crops.append(crop)

    return crops


# ------------ 3. Image Cropping: by num of rows and cols ------------
def smart_crop(img, num_rows, num_cols, min_overlap=0.3):
    """
    num_rows (int): 垂直方向分成几块
    num_cols (int): 水平方向分成几块
    min_overlap (float): 滑窗最小重叠比例 (0 ~ 1)
    """
    H, W = img.shape[:2]
    crops = []

    # 计算最大允许 stride，确保满足最小重叠
    max_stride_y = H / num_rows
    max_stride_x = W / num_cols

    max_crop_size_y = max_stride_y / (1 - min_overlap)
    max_crop_size_x = max_stride_x / (1 - min_overlap)

    # 最终正方形大小取 y/x 中较小者，确保不越界
    crop_size = int(min(max_crop_size_y, max_crop_size_x))

    # 步长取整，保证最后一块覆盖到边界
    stride_y = int((H - crop_size) / max(num_rows - 1, 1)) if num_rows > 1 else 0
    stride_x = int((W - crop_size) / max(num_cols - 1, 1)) if num_cols > 1 else 0

    # 滑窗裁图
    for row in range(num_rows):
        top = min(row * stride_y, H - crop_size)
        for col in range(num_cols):
            left = min(col * stride_x, W - crop_size)
            crop = img[top:top + crop_size, left:left + crop_size]
            crops.append(crop)

    return crops


def crop_images(
    image, filename, output_folder, top=0, left=0, crop_width=None, crop_height=None
):
    h, w = image.shape[:2]
    x = left
    y = top
    cw = crop_width if crop_width else w - x
    ch = crop_height if crop_height else h - y
    
    # 防止越界
    x_end = min(x + cw, w)
    y_end = min(y + ch, h)

    cropped = image[y:y_end, x:x_end]
    imwrite_unicode(os.path.join(output_folder, filename), cropped)


def do_resize(image, target_size=256):
    h, w = image.shape[:2]
    if w > target_size:
        img = resize_to_small(image, (target_size, target_size))
    elif w < target_size:
        img = resize_to_large(image, (target_size, target_size))
    else:
        img = image
    return img

if __name__ == "__main__":
    img_dir= "D:\Dataset-Drawing\明清彩绘本系列图像\\train_orig_512"
    save_dir = "D:\Dataset-Drawing\明清彩绘本系列图像\\train_orig_256"
    os.makedirs(save_dir, exist_ok=True)
    exts = ['.jpg', '.jpeg', '.png']
    files = os.listdir(img_dir)
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in exts]

    for idx, filename in enumerate(tqdm(image_files)):
        img_path = os.path.join(img_dir, filename)
        image = imread_unicode(img_path)

        # crop_images(
        #     image, filename, save_dir, 
        #     left=0, crop_width=0, top=300, crop_height=5000
        # )

        # # crop_list = crop_by_ratio(image, ratio_thresh=1.1)
        # crop_list = smart_crop(image, num_rows=3, num_cols=3, min_overlap=0.34)

        # for idx, item in enumerate(crop_list):
        #     new_filename = f"{os.path.splitext(filename)[0]}_{idx}.png"
        #     imwrite_unicode(os.path.join(save_dir, new_filename), item)
        
        img = do_resize(image, target_size=256)
        imwrite_unicode(os.path.join(save_dir, filename), img)