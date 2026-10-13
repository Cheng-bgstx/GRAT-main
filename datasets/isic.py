import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISIC2016(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Test'):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.transform_msk = transform_msk
        self.img_size = args.img_size

        csv_file = os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv')
        try:
            df = pd.read_csv(csv_file, encoding='gbk')
        except:
            df = pd.read_csv(csv_file)

        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):

        name = self.name_list[index]
        mask_name = self.label_list[index]

        img_path = os.path.join(self.data_path, name + '.jpg')
        msk_path = os.path.join(self.data_path, mask_name + '_Segmentation.png')

        if not name.endswith('.jpg'):
            img_path = os.path.join(self.data_path, name + '.jpg')
        else:
            img_path = os.path.join(self.data_path, name)

        if not str(mask_name).endswith('.png'):
            msk_path = os.path.join(self.data_path, str(mask_name) + '_Segmentation.png')
        else:
            msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        point_label = 1
        pt = np.array([self.img_size // 2, self.img_size // 2])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
        if self.transform_msk:
            torch.set_rng_state(state)
            mask = self.transform_msk(mask)
            if isinstance(mask, torch.Tensor):
                mask = (mask > 0.5).float()
        else:
            # 手动转换为 Tensor
            mask = np.array(mask)
            mask = torch.from_numpy(mask).float()

            # 如果 mask 是 0-255 的，为了保险可以归一化，或者保持原样
            # 攻击脚本里用的是 label < 1 来判断背景。
            # 如果 mask 前景是 255，背景是 0。 0 < 1 (True), 255 < 1 (False)。逻辑是通的。
            # 但为了规范，建议加上 unsqueeze 变成 [1, H, W] 格式
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)


        filename = os.path.splitext(os.path.basename(img_path))[0]
        image_meta_dict = {'filename_or_obj': filename}

        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }