import os
import shutil

def copy_and_rename_images(folder_path, copy_num):
    # 获取所有图像文件（假设是 jpg/png 格式）
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # 确保文件按照顺序处理

    for image in images:
        image_path = os.path.join(folder_path, image)
        output_index = 1  # 用于命名新文件

        for _ in range(copy_num):  # 复制 copy_num 次
            new_name = f"{os.path.splitext(image)[0]}-{output_index}{os.path.splitext(image)[1]}"
            new_path = os.path.join(folder_path, new_name)
            
            shutil.copy(image_path, new_path)  # 复制文件
            print(f"Copied {image} → {new_name}")

            output_index += 1  # 递增编号

# 使用示例
folder = "repaint\data\datasets\gt_keep_masks\\2-mask128-2"  # 替换为你的文件夹路径
copy_and_rename_images(folder, copy_num=2)  # 每张图像复制 2 次
