from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    针对一个数据集, 创建一个 (图像, 标签) 对生成器。

    返回值:  1. image_tensor, {"y": class_index}
            2. image_tensor, {}
    """
    # 1. 如果没有指定数据集的根目录，抛出异常；否则递归的获取所有图像文件
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)

    # 2. 如果使用了 class_cond 类别条件生成，建立类别和图像的映射索引
    classes = None
    if class_cond:
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    
    # 3. 构造自定义数据集
    dataset = ImageDataset(image_size, all_files, classes=classes)

    # 4. deterministic 为真，表示测试/验证，不打乱数据；为假，表示训练，打乱数据。
    loader_shuffle = not deterministic
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=loader_shuffle, 
        num_workers=1, drop_last=True
    )

    # 5. 生成器函数，可以无限循环产出数据 batch
    while True:
        yield from loader


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


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution

        # 多卡数据切片操作，单卡同样适用，保留
        self.local_images = image_paths[shard::num_shards]
        self.local_classes = None if classes is None else classes[shard::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # 1. 获取本进程对应的数据路径，通过 blobfile 读取文件内容，写入内存
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # 2. 缩小图片，每次减半，直到最短边小于2倍的 resolution
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        # 3. 计算 scale 因子，让最短边缩小到 resolution
        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        # 4. 中心裁剪成 resolution * resolution 的正方形
        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        
        # 5. 归一化到 [-1, 1]；如果标签存在，为图像加上标签
        arr = arr.astype(np.float32) / 127.5 - 1
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        # 6. np.transpose 将图片从 HWC 转化为 CHW 的 PyTorch 形式，返回图像和标签
        return np.transpose(arr, [2, 0, 1]), out_dict
