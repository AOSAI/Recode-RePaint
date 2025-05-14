import random
import os
from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

def load_data_yield(loader):
    while True:
        yield from loader

def load_data_inpa(
    *, gt_path=None, mask_path=None, batch_size, image_size, class_cond=False, 
    deterministic=False, random_crop=False, random_flip=True,
    return_dataloader=False, return_dict=False, max_len=None, 
    drop_last=True, conf=None, offset=0, **kwargs
):
    """
    针对一个数据集, 在 (图像、kwargs) 对上创建一个生成器。每个图像都是一个 NCHW 浮点张量。
    函数参数中的 * 表示：限制后面必须用关键字传参；**kwargs 表示收集多余的关键字参数到一个字典里。
    """
    # 1. expanduser 用来解析路径中的 ～ 符号，确保代码可以跨平台运行
    gt_dir = os.path.expanduser(gt_path)
    mask_dir = os.path.expanduser(mask_path)
    # 2. 递归获取图像、遮罩图像的文件路径；确保两种图像的数量是一致的。
    gt_paths = _list_image_files_recursively(gt_dir)
    mask_paths = _list_image_files_recursively(mask_dir)
    assert len(gt_paths) == len(mask_paths)

    # 不支持条件扩散生成，已弃用
    classes = None
    if class_cond:
        raise NotImplementedError()  

    # 3. 构造自定义数据集，返回值为3个键值对的字典
    # RePaint的定义：random_crop 随机裁剪从而扩增图像；random_flip 随机翻转图像。
    dataset = ImageDatasetInpa(
        image_size, gt_paths=gt_paths, mask_paths=mask_paths, classes=classes,
        shard=0, num_shards=1, random_crop=random_crop, random_flip=random_flip,
        return_dict=return_dict, max_len=max_len, conf=conf, offset=offset
    )

    # 4. deterministic 为真，表示测试/验证，不打乱数据；为假，表示训练，打乱数据。
    loader_shuffle = not deterministic
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=loader_shuffle, 
        num_workers=1, drop_last=drop_last
    )

    # 5. 采样时一般都返回 loader；训练的时候使用生成器函数无限采样 batch
    if return_dataloader:
        return loader
    else:
        return load_data_yield(loader)


# ------------ 递归获取图像的文件路径 ------------
def _list_image_files_recursively(data_dir):
    results = []
    # 类似 os.listdir()，返回当前目录下的文件名和子目录名。排序是为了顺序一致。
    for entry in sorted(bf.listdir(data_dir)):
        # 获取完整路径，拆分文件后缀
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        # 如果满足条件则写入 results，否则判断其是否为文件夹，是的话就递归调用当前函数
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDatasetInpa(Dataset):
    def __init__(
        self, resolution, gt_paths, mask_paths, classes=None,
        shard=0, num_shards=1, random_crop=False, random_flip=True,
        return_dict=False, max_len=None, conf=None, offset=0
    ):
        super().__init__()
        self.resolution = resolution

        # offset 用于跳过一些数据，在调试、断点恢复、手动数据划分等地方使用
        gt_paths = sorted(gt_paths)[offset:]
        mask_paths = sorted(mask_paths)[offset:]
        
        # 多卡数据切片操作，单卡同样适用
        self.local_gts = gt_paths[shard:][::num_shards]
        self.local_masks = mask_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.return_dict = return_dict
        self.max_len = max_len

    def __len__(self):
        if self.max_len is not None:
            return self.max_len
        return len(self.local_gts)

    def __getitem__(self, idx):
        # 1. 获取本进程对应的 图像、遮罩 路径，通过 blobfile 读取并写入内存
        gt_path = self.local_gts[idx]
        pil_gt = self.imread(gt_path)
        mask_path = self.local_masks[idx]
        pil_mask = self.imread(mask_path)

        # 2. 随机裁剪并扩充图像数量不实现；仅做原本的高质量缩放和中心裁剪
        if self.random_crop:
            raise NotImplementedError()
        else:
            arr_gt = center_crop_arr(pil_gt, self.resolution)
            arr_mask = center_crop_arr(pil_mask, self.resolution)

        # 3. 随机左右翻转
        if self.random_flip and random.random() < 0.5:
            arr_gt = arr_gt[:, ::-1]
            arr_mask = arr_mask[:, ::-1]

        # 4. 原始图像归一化到 [-1,1]，遮罩图像归一化到 [0,1]
        arr_gt = arr_gt.astype(np.float32) / 127.5 - 1
        arr_mask = arr_mask.astype(np.float32) / 255.0

        out_dict = {}  # 如果标签存在，为图像加上标签；不实现
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        # 5. 只允许返回字典，return_dict 必须为 true
        if self.return_dict:
            return {
                'GT': np.transpose(arr_gt, [2, 0, 1]),
                'GT_name': os.path.basename(gt_path),
                'gt_keep_mask': np.transpose(arr_mask, [2, 0, 1]),
            }
        else:
            raise NotImplementedError()

    # 1.1 读取图像，写入内存，标准化为 RGB 格式
    def imread(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image


# 2.1 图像的高质量缩放、中心裁剪；reducing_gap 参数需要 Pillow 版本大于 9.1.0
def center_crop_arr(pil_image, image_size):
    # 2.1.1 计算 scale 因子，让最短边缩小到 image_size
    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    pil_image = pil_image.resize(new_size, resample=Image.BICUBIC, reducing_gap=2.0)

    # 2.1.2 中心裁剪成 image_size * image_size 的正方形，并返回
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
