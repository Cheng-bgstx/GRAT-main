import torch
import os
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def pad_mask_to_square(mask, target_size):
    """Pad mask to square without normalizing pixel values."""
    h, w = mask.shape[-2:]
    padh = target_size - h
    padw = target_size - w
    if padh > 0 or padw > 0:
        mask = F.pad(mask, (0, padw, 0, padh), value=0)
    return mask


class SynapseDataset(torch.utils.data.Dataset):
    """2D Slice Dataset for Synapse multi-organ abdominal CT.

    Labels are class indices 0-8 (0=background, 1-8=organs).
    Returns NOT binarized — keeps multi-class labels as long tensors.
    """

    def __init__(self, data_root, list_file, transform=None, train=True,
                 sam_trans=None, split_ratio=0.8, loop=1):
        self.data_root = data_root
        self.transform = transform
        self.train = train
        self.sam_trans = sam_trans
        self.loop = loop

        with open(list_file, 'r') as f:
            all_slices = [line.strip() for line in f if line.strip()]

        train_slices, test_slices = train_test_split(
            all_slices, test_size=1 - split_ratio, random_state=42
        )

        if train:
            self.slices = train_slices
        else:
            self.slices = test_slices

        self.data_size = len(self.slices)
        print(f'Synapse {"train" if train else "test"} samples: {self.data_size}')

    def __getitem__(self, index):
        index = index % self.data_size
        slice_id = self.slices[index]

        npz_path = os.path.join(self.data_root, 'train_npz', f'{slice_id}.npz')
        data = np.load(npz_path)
        image = data['image'].astype(np.float32)   # (512, 512)
        label = data['label'].astype(np.int64)      # (512, 512), class indices 0-8

        # Single-channel CT → 3-channel
        image = np.stack([image] * 3, axis=-1)

        if self.transform:
            image, label = self.transform(image, label)

        # Ensure torch tensors & correct dtypes
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float().permute(2, 0, 1)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()
        # transforms.ToTensor converts label to float; fix back to long for class indices
        if isinstance(label, torch.Tensor) and label.dtype != torch.int64:
            label = label.round().long()

        original_size = tuple(image.shape[1:3])

        # Resize image to SAM target size
        image = self.sam_trans.apply_image_torch(image)
        image_size = tuple(image.shape[1:3])

        # Label: keep as class indices, resize to match image spatial size
        if label.shape != image.shape[1:]:
            label = F.interpolate(
                label.unsqueeze(0).unsqueeze(0).float(),
                size=image_size, mode='nearest'
            ).squeeze().long()

        # PREPROCESS image (normalize + pad), label (pad ONLY, NO normalize!)
        img = self.sam_trans.preprocess(image)
        label = pad_mask_to_square(label, self.sam_trans.target_length)

        return (img, label, torch.Tensor(original_size), torch.Tensor(image_size))

    def __len__(self):
        return self.data_size * self.loop


def get_synapse_dataset(args, sam_trans):
    data_root = args.get('synapse_root',
                         '/root/data2/fyc/fyc/GRAT-main/data/Synapse_preprocess')
    list_file = os.path.join(data_root, 'lists', 'train.txt')
    from dataset.tfs import get_synapse_transform
    transform_train, transform_test = get_synapse_transform(args)
    ds_train = SynapseDataset(data_root, list_file, transform=transform_train,
                              train=True, sam_trans=sam_trans, loop=3)
    ds_test = SynapseDataset(data_root, list_file, transform=transform_test,
                             train=False, sam_trans=sam_trans)
    return ds_train, ds_test
