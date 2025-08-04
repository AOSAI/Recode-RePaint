import cv2
import os
import numpy as np
import fitz  # PyMuPDF
import shutil
import random

# ------------ 1. Support CV2: If image path have chinese ------------
def imread_unicode(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ Image reading failed: {path}")
    return img

def imwrite_unicode(path, image):
    ext = os.path.splitext(path)[-1]  # 例如 .jpg
    result, encoded = cv2.imencode(ext, image)
    if result:
        encoded.tofile(path)
    else:
        raise IOError(f"❌ Image encoding failure, cannot be saved: {path}")
    

# ------------ 2. Extract images from pdf ------------
def extract_images_from_pdf(pdf_path, output_folder, flag=1, count=0):
    """
    pdf_path (str): PDF 文件路径
    output_folder (str): 输出图像保存路径
    flag (int): 命名标记模式
    count (int): 多 PDF 时的计数标号（flag==0 时生效）
    """
    os.makedirs(output_folder, exist_ok=True)
    saved_images = []

    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"[错误] 无法打开 PDF: {pdf_path}, 错误信息: {e}")
        return []

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        images = page.get_images(full=True)

        if not images:
            continue  # 没有图片，跳过该页

        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                if flag == 1:
                    image_name = f"page{page_number+1}_img{img_index+1}.{image_ext}"
                else:
                    image_name = f"pdf{count}_page{page_number+1}_img{img_index+1}.{image_ext}"

                image_path = os.path.join(output_folder, image_name)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                saved_images.append(image_path)

            except Exception as e:
                print(f"[警告] 无法提取第 {page_number+1} 页第 {img_index+1} 张图: {e}")

    return saved_images


def merge_multi_dir_images(
    source_folders, target_folder, image_extensions=(".jpg", ".png", ".jpeg")
):
    count = 1  # 统一命名编号

    # 遍历所有源文件夹
    for folder in source_folders:
        for filename in sorted(os.listdir(folder)):  # 排序保证顺序一致
            if filename.lower().endswith(image_extensions):
                old_path = os.path.join(folder, filename)
                new_filename = f"{count:04d}" + os.path.splitext(filename)[-1]  # 统一命名
                new_path = os.path.join(target_folder, new_filename)

                shutil.move(old_path, new_path)  # 移动文件
                print(f"Moved: {old_path} → {new_path}")

                count += 1

    print("✅ 所有图像已重命名并移动完成！")


# ------------ 4. 获取根路径下的所有子文件夹 (仅一层，不递归) ------------
def get_all_subfolders(root_dir):
    return [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]


# ------------ 5. 通过随机重命名的方式打乱顺序 ------------
def shuffle_and_rename(folder_path, seed=None, prefix='', ext='.png'):
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if seed is not None:
        random.seed(seed)
    random.shuffle(image_files)

    # 临时重命名防止冲突
    temp_names = {old: f"__temp_{i}{os.path.splitext(old)[1]}" for i, old in enumerate(image_files)}
    for old, temp in temp_names.items():
        os.rename(os.path.join(folder_path, old), os.path.join(folder_path, temp))

    # 重新命名为顺序编号：img_00001.jpg, img_00002.jpg ...
    for i, temp in enumerate(temp_names.values(), start=1):
        new_name = f"{prefix}{i:05d}{ext}"
        os.rename(os.path.join(folder_path, temp), os.path.join(folder_path, new_name))

    print(f"重命名完成，共处理 {len(image_files)} 张图像。")


if __name__ == "__main__":
    # pdf_path = "F:\数据集集合\明清彩绘本系列原本\已采用-超大画卷\仿宋院本金陵图卷.清.杨大章绘.纸本设色.清乾隆五十六年.台北故宫博物院藏.pdf"
    # output_dir = "D:\Dataset-Drawing\明清彩绘本系列图像\超大画卷\\0000"

    # images = extract_images_from_pdf(pdf_path, output_dir, flag=0, count=1)
    # print(f"成功提取 {len(images)} 张图片")
    
    # 自定义多个源文件夹路径
    # source_folders = [
    #     "D:\Dataset-Drawing\明清彩绘本系列图像\超大画卷\\民生图",
    #     "D:\Dataset-Drawing\明清彩绘本系列图像\超大画卷\\000",
    # ]
    # root_dir = "D:\Dataset-Drawing\明清彩绘本系列图像\\train_orig_02"
    # source_folders = get_all_subfolders(root_dir)

    # # 目标文件夹，确保其存在
    # target_folder = "D:\Dataset-Drawing\明清彩绘本系列图像\\train_orig"
    # os.makedirs(target_folder, exist_ok=True)
    # merge_multi_dir_images(source_folders, target_folder)

    shuffle_dir = "D:\Dataset-Drawing\明清彩绘本系列图像\\train_orig_512"
    shuffle_and_rename(shuffle_dir, seed=1206)