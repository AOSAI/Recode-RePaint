import os
from PIL import Image
from collections import defaultdict
from os.path import expanduser


# 将列表中的图像名称转换成 ext(PNG) 的后缀
def to_file_ext(img_names, ext):
    img_names_out = []
    for img_name in img_names:
        splits = img_name.split('.')
        assert len(splits) == 2, f"File name needs exactly one '.': {img_name}"
        img_names_out.append(splits[0] + '.' + ext)
    return img_names_out

    
# 将 nparray 图像数据保存为图片
def write_images(imgs, img_names, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for image_name, image in zip(img_names, imgs):
        out_path = os.path.join(dir_path, image_name)
        Image.fromarray(image).save(out_path)


class NoneDict(defaultdict):
    def __init__(self):
        # return_None 是一个容错设定。对于访问不到的键，.xxx 不报错，返回 None
        # 对于没有设置的键，即使用 ["xxx"] 访问也返回 None（defaultdict）
        super().__init__(self.return_None)

    @staticmethod
    def return_None():
        return None

    # 手动实现（conf.xxx）方式的递归属性查找。
    def __getattr__(self, attr):
        val = self.get(attr)
        if isinstance(val, dict) and not isinstance(val, NoneDict):
            val = NoneDict.recursive_convert(val)
            self[attr] = val  # 缓存起来，避免重复转换
        return val
    
    @staticmethod
    def recursive_convert(d):
        """递归将所有 dict 转为 NoneDict"""
        if isinstance(d, dict):
            nd = NoneDict()
            for k, v in d.items():
                nd[k] = NoneDict.recursive_convert(v)
            return nd
        else:
            return d


class Default_Conf(NoneDict):
    def __init__(self):
        pass

    # 3.1 获取 data -> eval 下的自定义 key 的名称
    def get_default_eval_name(self):
        candidates = self['data']['eval'].keys()
        if len(candidates) != 1:
            raise RuntimeError(
                f"Need exactly one candidate for {self.name}: {candidates}")
        return list(candidates)[0]

    # 3.2 从 image_datasets.py 中获取图像数据加载器
    def get_dataloader(self, dset='train', dsName=None, batch_size=None, return_dataset=False):
        batch_size = self.batch_size if batch_size is None else batch_size

        candidates = self['data'][dset]
        ds_conf = candidates[dsName].copy()

        # 检查 ds_conf 中有没有 'mask_loader' 这个 key，有就返回，没有就是默认的 False
        if ds_conf.get('mask_loader', False):
            from .image_datasets import load_data_inpa
            return load_data_inpa(**ds_conf, conf=self)
        else:
            raise NotImplementedError()

    # 4.7.1 保存 inpainting 相关的图像
    def eval_imswrite(
        self, srs=None, img_names=None, dset=None, name=None, ext='png', 
        lrs=None, gts=None, gt_keep_masks=None,
    ):
        img_names = to_file_ext(img_names, ext)  # 更改图像的后缀格式至 PNG
        paths = self['data'][dset][name]['paths']

        if srs is not None:
            sr_dir_path = expanduser(paths['srs'])
            write_images(srs, img_names, sr_dir_path)

        if gt_keep_masks is not None:
            mask_dir_path = expanduser(paths['gt_keep_masks'])
            write_images(gt_keep_masks, img_names, mask_dir_path)

        if gts is not None and paths.get("gts"):
            gt_dir_path = expanduser(paths.get("gts"))
            write_images(gts, img_names, gt_dir_path)

        if lrs is not None:
            lrs_dir_path = expanduser(paths['lrs'])
            write_images(lrs, img_names, lrs_dir_path)

