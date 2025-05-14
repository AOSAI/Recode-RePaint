from functools import lru_cache
import os
import torch
from PIL import Image

from collections import defaultdict
from os.path import isfile, expanduser

def imwrite(path=None, img=None):
    Image.fromarray(img).save(path)

def to_file_ext(img_names, ext):
    img_names_out = []
    for img_name in img_names:
        splits = img_name.split('.')
        if not len(splits) == 2:
            raise RuntimeError("File name needs exactly one '.':", img_name)
        img_names_out.append(splits[0] + '.' + ext)

    return img_names_out

def write_images(imgs, img_names, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    for image_name, image in zip(img_names, imgs):
        out_path = os.path.join(dir_path, image_name)
        imwrite(img=image, path=out_path)



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
        return self.get(attr)


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
            from image_datasets import load_data_inpa
            return load_data_inpa(**ds_conf, conf=self)
        else:
            raise NotImplementedError()

    def get_debug_variance_path(self):
        return os.path.expanduser(
            os.path.join(self.get_default_eval_conf()['paths']['root'], 'debug/debug_variance')
        )

    # 4.7.1 保存 inpainting 相关的图像
    def eval_imswrite(
        self, srs=None, img_names=None, dset=None, name=None, ext='png', 
        lrs=None, gts=None, gt_keep_masks=None, verify_same=True
    ):
        img_names = to_file_ext(img_names, ext)

        if dset is None:
            dset = self.get_default_eval_name()

        max_len = self['data'][dset][name].get('max_len')

        if srs is not None:
            sr_dir_path = expanduser(self['data'][dset][name]['paths']['srs'])
            write_images(srs, img_names, sr_dir_path)

        if gt_keep_masks is not None:
            mask_dir_path = expanduser(
                self['data'][dset][name]['paths']['gt_keep_masks'])
            write_images(gt_keep_masks, img_names, mask_dir_path)

        gts_path = self['data'][dset][name]['paths'].get('gts')
        if gts is not None and gts_path:
            gt_dir_path = expanduser(gts_path)
            write_images(gts, img_names, gt_dir_path)

        if lrs is not None:
            lrs_dir_path = expanduser(
                self['data'][dset][name]['paths']['lrs'])
            write_images(lrs, img_names, lrs_dir_path)

    def pget(self, name, default=None):
        if '.' in name:
            names = name.split('.')
        else:
            names = [name]

        sub_dict = self
        for name in names:
            sub_dict = sub_dict.get(name, default)

            if sub_dict == None:
                return default

        return sub_dict
