import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class REFUGE(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='none',
                 plane=False):
        self.data_path = data_path

        folder_path = os.path.join(data_path, mode + '-400')
        if os.path.exists(folder_path):
            self.subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        else:
            self.subfolders = []
            print(f"Warning: Directory {folder_path} does not exist.")

        self.mode = mode
        self.prompt = prompt
        self.img_size = args.img_size
        self.mask_size = args.img_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):
        point_label = 1

        """Get the images"""
        subfolder = self.subfolders[index]

        name = os.path.basename(subfolder)

        img_path = os.path.join(subfolder, name + '.jpg')

        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]
        multi_rater_disc_path = [os.path.join(subfolder, name + '_seg_disc_' + str(i) + '.png') for i in range(1, 8)]

        img = Image.open(img_path).convert('RGB')

        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]
        multi_rater_disc = [Image.open(path).convert('L') for path in multi_rater_disc_path]

        pt = np.array([0, 0], dtype=np.int32)

        mask = None
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            trans_op = self.transform_msk if self.transform_msk else self.transform
            if self.transform_msk:
                multi_rater_cup = [trans_op(single_rater) for single_rater in multi_rater_cup]
            else:
                multi_rater_cup = [torch.as_tensor((trans_op(single_rater) > 0.5).float(), dtype=torch.float32) for
                                   single_rater in multi_rater_cup]

            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)

            mask_cup = F.interpolate(multi_rater_cup, size=(self.mask_size, self.mask_size), mode='bilinear',
                                     align_corners=False).mean(dim=0)

            if self.transform_msk:
                multi_rater_disc = [trans_op(single_rater) for single_rater in multi_rater_disc]
            else:
                multi_rater_disc = [torch.as_tensor((trans_op(single_rater) > 0.5).float(), dtype=torch.float32) for
                                    single_rater in multi_rater_disc]
            multi_rater_disc = torch.stack(multi_rater_disc, dim=0)

            mask_disc = F.interpolate(multi_rater_disc, size=(self.mask_size, self.mask_size), mode='bilinear',
                                      align_corners=False).mean(dim=0)

            torch.set_rng_state(state)

            mask = torch.concat([mask_cup, mask_disc], dim=0)
        else:

            mask = torch.zeros((2, self.mask_size, self.mask_size))

        box_cup = [0, 0, 0, 0]

        image_meta_dict = {'filename_or_obj': name}

        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'box': box_cup,
            'image_meta_dict': image_meta_dict,
        }