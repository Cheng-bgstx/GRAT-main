# datasets/dataset_npz.py

import os
import numpy as np
from torch.utils.data import Dataset


class NPZ_Dataset(Dataset):
    def __init__(self, base_dir, split='test'):
        self.base_dir = base_dir
        self.sample_list = []
        filenames = sorted([f for f in os.listdir(base_dir) if f.endswith('.npz')])
        for filename in filenames:
            self.sample_list.append(os.path.join(base_dir, filename))

        print(f"总共找到 {len(self.sample_list)} 个 {split} 数据")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        npz_path = self.sample_list[idx]
        case_name = os.path.basename(npz_path).replace('.npz', '')

        data = np.load(npz_path)
        image = data['imgs']  # (num_slices, H, W)
        label = data['gts']  # (num_slices, H, W)
        # 转换为 PyTorch 需要的 float 类型，并增加通道维度
        # test_single_volume 和 attack 脚本内部会处理3通道复制
        image = image.astype(np.float32)

        sample = {
            'image': image,
            'label': label,
            'case_name': case_name
        }

        return sample