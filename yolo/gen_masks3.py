import os
import shutil
import cv2
import numpy as np
from PIL import Image
from skimage import measure
import json

def copy_and_pullback(person_dir, repainted_dir, merged_dir):
    os.makedirs(merged_dir, exist_ok=True)

    for filename in os.listdir(person_dir):
        if filename.endswith(".png"):
            base = os.path.splitext(filename)[0].replace("-mask", "")
            repaired_path = os.path.join(repainted_dir, base + ".png")
            save_path = os.path.join(merged_dir, base + ".png")

            if os.path.exists(repaired_path):
                shutil.copy(repaired_path, save_path)
                print(f"✅ Copied: {base}.png")
            else:
                print(f"❌ Not found: {base}.png")

def overlay_person_on_repainted(person_rgba_path, repainted_rgb_path, save_path=None):
    # 读取图像
    person = cv2.imread(person_rgba_path, cv2.IMREAD_UNCHANGED)  # RGBA
    background = cv2.imread(repainted_rgb_path)  # RGB

    assert person.shape[:2] == background.shape[:2], "Image size mismatch"
    assert person.shape[2] == 4, "Person image must have alpha channel"

    # 拆分 RGBA 通道
    b, g, r, a = cv2.split(person)
    alpha = a.astype(float) / 255.0  # 透明度 [0,1]
    alpha = np.expand_dims(alpha, axis=2)

    # 合成图像
    foreground = cv2.merge([b, g, r]).astype(float)
    background = background.astype(float)

    blended = foreground * alpha + background * (1 - alpha)
    blended = blended.astype(np.uint8)

    # 保存或返回
    if save_path:
        cv2.imwrite(save_path, blended)
    return blended


def extract_effective_mask(image_path, save_path1):
    """
    从 RGBA 图像中提取“有效区域内”的纯黑色像素区域作为 mask。
    - 只有 alpha ≠ 0 的像素才参与判断
    - RGB == (0,0,0) 被视为缺失区域（mask）

    返回二值图像：0=缺失，255=正常
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 读取 RGBA 图像
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    if img.shape[2] != 4:
        raise ValueError(f"Expected RGBA image, got shape {img.shape}")

    rgb = img[:, :, :3]
    alpha = img[:, :, 3]

    # 仅在 alpha ≠ 0 的地方判断 RGB 是否为黑色
    black_region = np.all(rgb == 0, axis=2)
    visible_region = alpha != 0
    mask = np.where(visible_region & black_region, 0, 255).astype(np.uint8)

    cv2.imwrite(save_path1, mask)


def batch_overlay(person_dir, repainted_dir, save_dir, new_masks):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(new_masks, exist_ok=True)
    person_files = sorted(os.listdir(person_dir))

    for filename in tqdm(person_files, desc="Overlaying images"):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        base_name = filename.replace("-mask", "")
        person_path = os.path.join(person_dir, filename)
        repainted_path = os.path.join(repainted_dir, base_name)
        save_path = os.path.join(save_dir, base_name)

        save_path1 = os.path.join(new_masks, base_name)
        extract_effective_mask(person_path, save_path1)

        if not os.path.exists(repainted_path):
            print(f"[!] Skipped: {base_name} not found in repainted_dir")
            continue

        try:
            result = overlay_person_on_repainted(person_path, repainted_path)
            cv2.imwrite(save_path, result)
        except Exception as e:
            print(f"[!] Failed on {base_name}: {e}")
    

def img_mask_128_process(mask_path, img_path, mask128_path, img128_path, filename, json_file):
    # 加载图像，转换为 numpy 数组
    mask_img = Image.open(mask_path).convert("L")
    orig_img = Image.open(img_path).convert("RGB")
    mask_np = np.array(mask_img)
    
    # 获取非零区域坐标，如果没有遮罩则跳过
    ys, xs = np.where(mask_np < 128)  # 黑色像素位置
    if len(xs) == 0 or len(ys) == 0:
        return 0

    # 连通域检测（8连通）
    mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask_cv, 128, 255, cv2.THRESH_BINARY_INV)
    labels = measure.label(binary_mask, connectivity=2)
    regions = measure.regionprops(labels)

    if len(regions) == 0:
        return False  # 无连通区域，退出

    selected_regions = []
    selected_coords = []
    max_crop_area = 128 * 128
    max_mask_area = int(max_crop_area * 0.25)
    total_mask_area = 0

    for region in regions:
        region_coords = region.coords
        region_area = region.area

        # 尝试合并新区域
        if len(selected_coords) > 0:
            tentative_coords = np.vstack([selected_coords, region_coords])
        else:
            tentative_coords = region_coords
        
        minr, minc = np.min(tentative_coords, axis=0)
        maxr, maxc = np.max(tentative_coords, axis=0)
        width, height = maxc - minc, maxr - minr

        # 新区域 + 现有总和，是否仍满足边界与面积限制
        if width <= 128 and height <= 128 and (total_mask_area + region_area) <= max_mask_area:
            selected_regions.append(region)
            selected_coords = tentative_coords
            total_mask_area += region_area
        else:
            continue

    # 如果没有选中任何合格区域 → 尝试从最大区域中截取中心块
    if len(selected_coords) == 0:
             # 按区域面积排序，选择最大区域
        largest_region = max(regions, key=lambda r: r.area)
        region_coords = largest_region.coords
        region_area = largest_region.area

        if region_area == 0:
            return False  # 异常区域跳过

        # 计算该区域的边界框中心
        minr, minc = np.min(region_coords, axis=0)
        maxr, maxc = np.max(region_coords, axis=0)
        center_x = (minc + maxc) // 2
        center_y = (minr + maxr) // 2

        # 计算裁剪区域（128x128）
        half_size = 64
        left = max(center_x - half_size, 0)
        upper = max(center_y - half_size, 0)
        right = min(left + 128, 256)
        lower = min(upper + 128, 256)
        if right - left < 128:
            left = right - 128
        if lower - upper < 128:
            upper = lower - 128

        # 创建新 mask，只保留该区域内、且在裁剪框内的像素
        isolated_mask = np.ones_like(mask_np) * 255
        for y, x in region_coords:
            if upper <= y < lower and left <= x < right:
                isolated_mask[y, x] = 0  # 黑色

        # 覆盖 selected_coords，供后续通用逻辑使用
        selected_coords = region_coords
    else:
        # 裁剪框坐标
        minr, minc = np.min(selected_coords, axis=0)
        maxr, maxc = np.max(selected_coords, axis=0)
        center_x = (minc + maxc) // 2
        center_y = (minr + maxr) // 2

        # 计算裁剪区域（128x128）
        half_size = 64
        left = max(center_x - half_size, 0)
        upper = max(center_y - half_size, 0)
        right = min(left + 128, 256)
        lower = min(upper + 128, 256)
        if right - left < 128:
            left = right - 128
        if lower - upper < 128:
            upper = lower - 128

        # 创建新 mask，只保留 selected_coords 中、且在裁剪框内的像素
        isolated_mask = np.ones_like(mask_np) * 255
        for y, x in selected_coords:
            if upper <= y < lower and left <= x < right:
                isolated_mask[y, x] = 0  # 黑色
    
    # 裁剪新图
    cropped_mask_array = isolated_mask[upper:lower, left:right]
    cropped_mask_img = Image.fromarray(cropped_mask_array)
    cropped_img = orig_img.crop((left, upper, right, lower))

    # 保存裁剪图像
    cropped_mask_img.save(mask128_path)
    cropped_img.save(img128_path)

    # 在原始 mask 上白化 selected_coords 中在框内的像素
    for y, x in selected_coords:
        if upper <= y < lower and left <= x < right:
            mask_np[y, x] = 255
    updated_mask_img = Image.fromarray(mask_np)
    updated_mask_img.save(mask_path)

    # 写入 JSON 文件
    record = {
        filename: {
            "crop_box": [int(left), int(upper), int(right), int(lower)],
            "128_mask_path": mask128_path,
            "128_img_path": img128_path,
            "256_mask_path": mask_path,
            "256_img_path": img_path,
        }
    }

    # 读取已有记录（如有）
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # 更新字典并写回
    data.update(record)

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    return True  # 成功完成一轮处理

def from_256_to_128(img128_dir, mask128_dir, merged_dir, new_masks, record_file):
    os.makedirs(img128_dir, exist_ok=True)
    os.makedirs(mask128_dir, exist_ok=True)

    exts = ['.jpg', '.jpeg', '.png']
    files1 = os.listdir(merged_dir)
    files2 = os.listdir(new_masks)
    image_files = [f for f in files1 if os.path.splitext(f)[1].lower() in exts]
    mask_files = [f for f in files2 if os.path.splitext(f)[1].lower() in exts]
    assert len(image_files) == len(mask_files), "images and masks should have the same length"

    for idx, filename in enumerate(tqdm(image_files)):
        img_path = os.path.join(merged_dir, filename)
        mask_path = os.path.join(new_masks, filename)
        filename1 = os.path.splitext(filename)[0]

        img128_path = os.path.join(img128_dir, filename)
        mask128_path = os.path.join(mask128_dir, filename)

        img_mask_128_process(mask_path, img_path, mask128_path, img128_path, filename1, record_file)


def paste_inpainted_regions(json_path, inpainted_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    for key, info in data.items():
        crop_box = info["crop_box"]
        x1, y1, x2, y2 = crop_box

        orig_img_path = info["256_img_path"]
        inpainted_img_path = os.path.join(inpainted_dir, f"{key}.png")

        if not os.path.exists(inpainted_img_path):
            print(f"[跳过] 修复图不存在: {inpainted_img_path}")
            continue

        # 读取图像
        orig_img = Image.open(orig_img_path).convert("RGB")
        inpainted_img = Image.open(inpainted_img_path).convert("RGB")

        # 尺寸检查
        if inpainted_img.size != (x2 - x1, y2 - y1):
            print(f"[跳过] 修复图尺寸不匹配 {key}: {inpainted_img.size} vs crop_box {crop_box}")
            continue

        # 贴回
        orig_img.paste(inpainted_img, (x1, y1))

        # 保存
        save_path = os.path.join(save_dir, f"{key}.png")
        orig_img.save(save_path)
        print(f"[完成] {key} 贴回图像保存至: {save_path}")


import os

def filter_folder_by_reference(folder_a, folder_b, match_extension=False, dry_run=False):
    """
    - folder_a: 用于筛选参考的文件夹路径
    - folder_b: 要删除多余图像的文件夹路径
    - match_extension: 是否按“完整文件名（含扩展名）”匹配（默认只比对文件名）
    - dry_run: 是否只打印将被删除的文件（True 则不会实际删除）
    """
    if not os.path.isdir(folder_a) or not os.path.isdir(folder_b):
        raise ValueError("两个路径都必须是存在的文件夹")

    if match_extension:
        keep_names = set(os.listdir(folder_a))  # 完整文件名
    else:
        keep_names = {os.path.splitext(f)[0] for f in os.listdir(folder_a) if os.path.isfile(os.path.join(folder_a, f))}

    deleted_files = []

    for file in os.listdir(folder_b):
        file_path = os.path.join(folder_b, file)
        if not os.path.isfile(file_path):
            continue

        name_key = file if match_extension else os.path.splitext(file)[0]
        if name_key not in keep_names:
            if dry_run:
                print(f"[Dry Run] 将删除: {file_path}")
            else:
                os.remove(file_path)
                print(f"已删除: {file_path}")
            deleted_files.append(file_path)

    print(f"\n共删除 {len(deleted_files)} 个文件。" if not dry_run else f"\n共标记 {len(deleted_files)} 个文件准备删除。")


from tqdm import tqdm
if __name__ == "__main__":
    person_dir = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\2-person"
    repainted_dir = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\3-inpainted_xt"
    selected_dir = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\4-img"
    merged_dir = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\4-img-2"
    new_masks = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\4-mask"
    
    # # 1.2. 筛选图像、pushback、制作mask
    # copy_and_pullback(person_dir, repainted_dir, selected_dir)
    # batch_overlay(person_dir, selected_dir, merged_dir, new_masks)

    # 3. 128 像素 img 与 mask 的处理
    img128_dir = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\5-img128-2"
    mask128_dir = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\5-mask128-2"
    record_file = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\5-cropped_info-2.json"
    # from_256_to_128(img128_dir, mask128_dir, merged_dir, new_masks, record_file)

    # 4. 128 图像贴回 256 中
    inpainted_dir = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle-1\\3-inpainted_xt"
    save_dir = "D:\CodeReproduction\Recode-RePaint\\repaint\log\\3m_randcircle\\4-img-3"
    paste_inpainted_regions(record_file, inpainted_dir, save_dir)
    filter_folder_by_reference(save_dir, new_masks, match_extension=False, dry_run=False)
    