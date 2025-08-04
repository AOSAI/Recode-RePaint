from ultralytics.models import YOLO
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from segment_anything import sam_model_registry, SamPredictor
import json


def sam_mask(image, filtered_boxes, predictor, gt_masked=None):
    mask_list = []
    orig_list = []

    for idx, box in enumerate(filtered_boxes):
        input_box = np.array(box)

        # 4.1. SAM 推理, 选择最优 mask
        masks, scores, logits = predictor.predict(
            box=input_box, multimask_output=True  # 输出多个候选结果
        )
        best_mask = masks[np.argmax(scores)]

        # 4.2 将 mask 二值化（黑白，0和255）
        mask_img = (best_mask.astype(np.uint8)) * 255

        # 4.3 定义膨胀核（通常用 5x5 或 7x7 核），执行膨胀，生成 0/255 二值 mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(mask_img, kernel, iterations=1)
        mask_img = np.where(dilated_mask, 0, 255).astype(np.uint8)

        # 4.4 生成 RGBA 人像，无缺/有缺
        if gt_masked == None:
            rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)   # BGR → BGRA
            rgba[:, :, 3] = dilated_mask  # 替换 alpha 通道
        else:
            img = cv2.imread(gt_masked)
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = dilated_mask  # 只保留 SAM 掩码对应的区域，其它透明

        mask_list.append(mask_img)
        orig_list.append(rgba)

    return mask_list, orig_list


def ensure_2d_mask(mask):
    mask = np.squeeze(mask)
    assert mask.ndim == 2, f"Expected 2D mask, got shape {mask.shape}"
    return mask.astype(np.uint8)


def save_merged_mask(mask_list, mask, save_dir, filename1):
    if len(mask_list) == 0:
        cv2.imwrite(os.path.join(save_dir, f"{filename1}.png"), mask)
        return 0

    mask = ensure_2d_mask(mask)
    mask_stack = np.stack(mask_list)         # shape [N, H, W]
    mask_union = np.min(mask_stack, axis=0)  # 取所有 mask 的交集区域为 0（黑），其余为 255

    assert mask.ndim == 2, f"mask must be 2D but got shape {mask.shape}"
    assert mask_union.shape == mask.shape, "Shape mismatch between mask and mask_union"
    
    merged_mask = np.minimum(mask, mask_union)  # 叠加mask
    cv2.imwrite(os.path.join(save_dir, f"{filename1}.png"), merged_mask)

    mask_union1 = np.max(mask_stack, axis=0)
    merged_mask1 = np.where((mask == 0) & (mask_union1 == 0), 0, 255).astype(np.uint8)
    merged_subdir = os.path.join(save_dir, "merged_mask")
    os.makedirs(merged_subdir, exist_ok=True)
    cv2.imwrite(os.path.join(merged_subdir, f"{filename1}.png"), merged_mask1)


def save_orig_person(orig_list, save_dir, filename1):
    merged_rgba = np.zeros_like(orig_list[0])  # 全透明背景图
    for rgba in orig_list:
        # 只保留当前 rgba 中 alpha>0 的区域
        mask = rgba[:, :, 3] > 0
        for c in range(4):  # 包括 RGB 和 A
            merged_rgba[:, :, c] = np.where(mask, rgba[:, :, c], merged_rgba[:, :, c])
        
    cv2.imwrite(os.path.join(save_dir, f"{filename1}-mask.png"), merged_rgba)


def filter_boxes_by_mask_overlap(boxes, mask):
    """
    boxes: [N, 4] numpy array, xyxy
    mask: [H, W] numpy array, binary (0 for masked/black, 255 for normal)
    只保留那些和黑色区域有一定“高度交集”的 box
    """
    keep_boxes = []
    for box in boxes:
        x0, y0, x1, y1 = map(int, box)
        
        # 边界限制（避免越界）
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(mask.shape[1], x1)
        y1 = min(mask.shape[0], y1)
        box_mask = mask[y0:y1, x0:x1]

        if np.any(box_mask == 0):
            keep_boxes.append(box)

    return np.array(keep_boxes)


def load_rects_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = []
    for shape in data.get('shapes', []):
        if shape.get('label') != 'person' or shape.get('shape_type') != 'rectangle':
            continue

        points = shape['points']
        # x1y1 是左上，x2y2 是右下
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)

        rects.append([x1, y1, x2, y2])
    
    return rects  # list of [x1, y1, x2, y2]


def detect(
    img_path, mask_path, save_dir1, save_dir2, filename1, 
    masked_path=None, json_path=None
):
    # 1. 加载图像, 加载保存目录
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    if json_path == None:
        yolo = YOLO("yolo/myProject/runs/train/exp5/weights/best.pt")
        results = yolo.predict(
            source=image,
            imgsz=640,
            conf=0.25,  # 最小置信阈值
            iou=0.5,  # NMS 的 IoU 阈值
            max_det=500,  # 最大检测次数
            device="0",
            classes=[0],  # 指定 ID 类别过滤，0 为 person
            # save=True,
            # project=save_dir,
            show_labels=False,  # 是否显示检测框的标签
            verbose=False,  # ← 关掉控制台输出
        )
        yolo_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        filtered_boxes = filter_boxes_by_mask_overlap(yolo_boxes, mask)
    else:
        filtered_boxes = np.array(load_rects_from_json(json_path))

    if len(filtered_boxes) == 0:
        save_merged_mask([], mask, save_dir2, filename1)
        return 0

    # 3. 加载 sam 模型
    sam_checkpoint = "yolo/myProject/sam_vit_b.pth"
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    sam.to(device)
    predictor = SamPredictor(sam)  # 将 SAM 模型封装进一个 预测器对象，方便后续推理
    predictor.set_image(image)  # 将要处理的图像送入预测器的编码器中
    mask_list, orig_list = sam_mask(image, filtered_boxes, predictor, masked_path)
    save_orig_person(orig_list, save_dir1, filename1)
    save_merged_mask(mask_list, mask, save_dir2, filename1)

from tqdm import tqdm
if __name__ == "__main__":
    img_dir= "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\1-gt"
    masked_dir = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\1-gt_masked"
    mask_dir = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\1-gt_keep_mask"
    save_dir1 = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\2-person"
    save_dir2 = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\2-mask"

    exts = ['.jpg', '.jpeg', '.png']
    files1 = os.listdir(img_dir)
    files2 = os.listdir(mask_dir)
    image_files = [f for f in files1 if os.path.splitext(f)[1].lower() in exts]
    mask_files = [f for f in files2 if os.path.splitext(f)[1].lower() in exts]
    assert len(image_files) == len(mask_files), "images and masks should have the same length"

    for idx, filename in enumerate(tqdm(image_files)):
        img_path = os.path.join(img_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        filename1 = os.path.splitext(filename)[0]

        masked_path = os.path.join(masked_dir, filename)
        masked_path = masked_path if os.path.exists(masked_path) else None
        # 判断是否存在对应的 .json 文件
        json_file = f"{filename1}.json"
        json_path = os.path.join(img_dir, json_file)
        json_path = json_path if os.path.exists(json_path) else None
        
        # cur_prefix = f"{idx}"  # 生成一个前缀，方便区分每张图像
        detect(img_path, mask_path, save_dir1, save_dir2, filename1, masked_path, json_path)