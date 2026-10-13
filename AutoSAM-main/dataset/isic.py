import torch
import os
import numpy as np
import cv2
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def cv2_loader(path, is_mask):
    if is_mask:
        img = cv2.imread(path, 0)
        img[img > 0] = 1
    else:
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img


def pad_mask_to_square(mask, target_size):
    """Pad binary/long mask to square without normalizing pixel values."""
    h, w = mask.shape[-2:]
    padh = target_size - h
    padw = target_size - w
    if padh > 0 or padw > 0:
        mask = F.pad(mask, (0, padw, 0, padh), value=0)
    return mask


class ISICDataset(torch.utils.data.Dataset):
    """2D Dataset for ISIC 2016 skin lesion segmentation (binary)."""

    def __init__(self, image_root, gt_root, transform=None, train=True,
                 sam_trans=None, split_ratio=0.8, loop=1):
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transform
        self.train = train
        self.sam_trans = sam_trans
        self.loop = loop

        all_images = sorted([
            f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        all_masks = []
        for img_file in all_images:
            base = os.path.splitext(img_file)[0]
            mask_file = base + '_Segmentation.png'
            all_masks.append(mask_file)

        valid_pairs = []
        for img_file, mask_file in zip(all_images, all_masks):
            if os.path.exists(os.path.join(gt_root, mask_file)):
                valid_pairs.append((img_file, mask_file))

        train_pairs, test_pairs = train_test_split(
            valid_pairs, test_size=1 - split_ratio, random_state=42
        )

        if train:
            self.data_pairs = train_pairs
        else:
            self.data_pairs = test_pairs

        self.data_size = len(self.data_pairs)
        print(f'ISIC {"train" if train else "test"} samples: {self.data_size}')

    def __getitem__(self, index):
        index = index % self.data_size
        img_file, mask_file = self.data_pairs[index]

        img = cv2_loader(os.path.join(self.image_root, img_file), is_mask=False)
        mask = cv2_loader(os.path.join(self.gt_root, mask_file), is_mask=True)

        if self.transform:
            img, mask = self.transform(img, mask)

        # Ensure torch tensors
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            mask = torch.from_numpy(mask).float()

        original_size = tuple(img.shape[1:3])

        # Resize both to SAM target size
        img = self.sam_trans.apply_image_torch(img)
        mask = self.sam_trans.apply_image_torch(mask)
        mask = (mask > 0.5).float()

        image_size = tuple(img.shape[1:3])

        # PREPROCESS image (normalize + pad), mask (pad ONLY, no normalize!)
        img = self.sam_trans.preprocess(img)
        mask = pad_mask_to_square(mask, self.sam_trans.target_length)

        return (img, mask, torch.Tensor(original_size), torch.Tensor(image_size))

    def __len__(self):
        return self.data_size * self.loop


def get_isic_dataset(args, sam_trans):
    image_root = args.get('isic_image_root',
                          '/root/data2/fyc/fyc/GRAT-main/data/ISIC/ISBI2016_ISIC_Part1_Training_Data')
    gt_root = args.get('isic_gt_root',
                       '/root/data2/fyc/fyc/GRAT-main/data/ISIC/ISBI2016_ISIC_Part1_Training_GroundTruth')
    from dataset.tfs import get_isic_transform
    transform_train, transform_test = get_isic_transform(args)
    ds_train = ISICDataset(image_root, gt_root, transform=transform_train,
                           train=True, sam_trans=sam_trans, loop=5)
    ds_test = ISICDataset(image_root, gt_root, transform=transform_test,
                          train=False, sam_trans=sam_trans)
    return ds_train, ds_test
