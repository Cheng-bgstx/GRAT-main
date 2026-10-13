import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

class ResizeAndPad:
    """
    处理可变分辨率的图像和标签。
    1. 保持长宽比，将图像的最长边缩放到目标尺寸。
    2. 在较短的边上进行填充，使其成为一个正方形。
    """
    def __init__(self, output_size, fill=0, label_fill=0):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.fill = fill
        self.label_fill = label_fill

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 获取原始尺寸
        w, h = image.size
        target_h, target_w = self.output_size

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # 缩放图像和标签
        # 使用高质量的重采样方法处理图像，使用最近邻方法处理标签
        image = image.resize((new_w, new_h), Image.BICUBIC)
        label = label.resize((new_w, new_h), Image.NEAREST)

        # 计算需要填充的量
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))

        # 进行填充
        image = TF.pad(image, padding, fill=self.fill, padding_mode='constant')
        label = TF.pad(label, padding, fill=self.label_fill, padding_mode='constant')

        return {'image': image, 'label': label}

class ToTensorAndNormalize:
    """
    将PIL Image或numpy ndarray转换为Tensor，并进行归一化。
    """
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 转换为Tensor
        # 图像: 从 (H, W, C) numpy array 或 PIL Image -> (C, H, W) tensor
        image = TF.to_tensor(image) # 自动将[0, 255]范围的PIL Image或uint8 numpy转为[0.0, 1.0]的Float Tensor

        # 标签: PIL Image (L mode) -> Tensor
        # 注意：标签不应被归一化
        label = torch.from_numpy(np.array(label, dtype=np.int64))

        return {'image': image, 'label': label}