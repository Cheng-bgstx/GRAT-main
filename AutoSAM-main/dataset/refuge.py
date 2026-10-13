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
    """Pad mask to square without normalizing pixel values."""
    h, w = mask.shape[-2:]
    padh = target_size - h
    padw = target_size - w
    if padh > 0 or padw > 0:
        mask = F.pad(mask, (0, padw, 0, padh), value=0)
    return mask


class REFUGEDataset(torch.utils.data.Dataset):
    """2D Dataset for REFUGE fundus image segmentation.

    Returns 2-channel mask: [cup, disc] (binary per channel).
    Matches Medical-SAM-Adapter's REFUGE handling.
    """

    def __init__(self, root, transform=None, train=True, sam_trans=None,
                 split_ratio=0.8, loop=1, rater_id=1):
        self.root = root
        self.transform = transform
        self.train = train
        self.sam_trans = sam_trans
        self.loop = loop
        self.rater_id = rater_id

        subject_dirs = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')
        ])

        valid_subjects = []
        for subj in subject_dirs:
            subj_path = os.path.join(root, subj)
            img_path = os.path.join(subj_path, f'{subj}.jpg')
            disc_png = os.path.join(subj_path, f'{subj}_seg_disc_{rater_id}.png')
            cup_png = os.path.join(subj_path, f'{subj}_seg_cup_{rater_id}.png')
            if os.path.exists(img_path) and os.path.exists(disc_png) and os.path.exists(cup_png):
                valid_subjects.append(subj)

        train_subjects, test_subjects = train_test_split(
            valid_subjects, test_size=1 - split_ratio, random_state=42
        )

        if train:
            self.subjects = train_subjects
        else:
            self.subjects = test_subjects

        self.data_size = len(self.subjects)
        print(f'REFUGE {"train" if train else "test"} samples: {self.data_size}')

    def __getitem__(self, index):
        index = index % self.data_size
        subj = self.subjects[index]
        subj_path = os.path.join(self.root, subj)

        img = cv2_loader(os.path.join(subj_path, f'{subj}.jpg'), is_mask=False)

        cup_raw = cv2.imread(os.path.join(subj_path, f'{subj}_seg_cup_{self.rater_id}.png'), 0)
        disc_raw = cv2.imread(os.path.join(subj_path, f'{subj}_seg_disc_{self.rater_id}.png'), 0)

        cup_mask = (cup_raw >= 128).astype(np.float32)
        disc_mask = (disc_raw >= 128).astype(np.float32)
        mask = np.stack([cup_mask, disc_mask], axis=0)  # (2, H, W) — PyTorch convention

        if self.transform:
            img, mask = self.transform(img, mask)

        # Ensure torch tensors
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            mask = torch.from_numpy(mask).float().permute(2, 0, 1)

        original_size = tuple(img.shape[1:3])

        # Resize both
        img = self.sam_trans.apply_image_torch(img)
        mask = self.sam_trans.apply_image_torch(mask)
        mask = (mask > 0.5).float()

        image_size = tuple(img.shape[1:3])

        # PREPROCESS image (normalize + pad), mask (pad ONLY)
        img = self.sam_trans.preprocess(img)
        mask = pad_mask_to_square(mask, self.sam_trans.target_length)

        return (img, mask, torch.Tensor(original_size), torch.Tensor(image_size))

    def __len__(self):
        return self.data_size * self.loop


def get_refuge_dataset(args, sam_trans):
    root = args.get('refuge_root',
                    '/root/data2/fyc/fyc/GRAT-main/data/REFUGE-Multirater/Training-400')
    rater_id = args.get('refuge_rater_id', 1)
    from dataset.tfs import get_refuge_transform
    transform_train, transform_test = get_refuge_transform(args)
    ds_train = REFUGEDataset(root, transform=transform_train, train=True,
                             sam_trans=sam_trans, loop=5, rater_id=rater_id)
    ds_test = REFUGEDataset(root, transform=transform_test, train=False,
                            sam_trans=sam_trans, rater_id=rater_id)
    return ds_train, ds_test
