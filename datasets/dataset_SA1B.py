import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import random
from typing import Optional, Dict, List
import pycocotools.mask as mask_utils
from typing import Dict, Tuple


class SA1BDataset(Dataset):
    """
    SA-1B (Segment Anything 1 Billion) dataset loader
    """

    def __init__(
            self,
            root_dir: str,
            img_size: int = 512,
            num_samples: Optional[int] = None,
            load_masks: bool = False,
            seed: int = 42
    ):
        """
        Args:
            root_dir: Path to SA-1B dataset root directory
            img_size: Target image size for resizing
            num_samples: Number of samples to use (None for all)
            load_masks: Whether to load segmentation masks
            seed: Random seed for sample selection
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.load_masks = load_masks

        # SA-1B directory structure
        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_dir = os.path.join(root_dir, 'annotations')

        # Get all image files
        self.image_files = []
        self.annotation_files = []

        # Check if using SA-1B structure with subdirectories
        if os.path.exists(self.images_dir):
            # SA-1B uses a specific directory structure with sa_XXXXX folders
            for subdir in sorted(os.listdir(self.images_dir)):
                if subdir.startswith('sa_'):
                    subdir_path = os.path.join(self.images_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for img_name in os.listdir(subdir_path):
                            if img_name.endswith('.jpg'):
                                img_path = os.path.join(subdir_path, img_name)
                                self.image_files.append(img_path)

                                # Find corresponding annotation
                                ann_name = img_name.replace('.jpg', '.json')
                                ann_subdir = subdir  # Same structure for annotations
                                ann_path = os.path.join(self.annotations_dir, ann_subdir, ann_name)
                                if os.path.exists(ann_path):
                                    self.annotation_files.append(ann_path)
                                else:
                                    self.annotation_files.append(None)
        else:
            # Fallback: flat directory structure
            for img_name in os.listdir(root_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_files.append(os.path.join(root_dir, img_name))

                    # Try to find annotation
                    ann_name = os.path.splitext(img_name)[0] + '.json'
                    ann_path = os.path.join(root_dir, ann_name)
                    if os.path.exists(ann_path):
                        self.annotation_files.append(ann_path)
                    else:
                        self.annotation_files.append(None)

        # Sample if requested
        if num_samples is not None and num_samples < len(self.image_files):
            random.seed(seed)
            indices = random.sample(range(len(self.image_files)), num_samples)
            self.image_files = [self.image_files[i] for i in indices]
            self.annotation_files = [self.annotation_files[i] for i in indices]

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        print(f"Loaded {len(self.image_files)} images from SA-1B dataset")
        if self.load_masks:
            valid_annotations = sum(1 for ann in self.annotation_files if ann is not None)
            print(f"Found {valid_annotations} valid annotation files")

    def __len__(self):
        return len(self.image_files)

    def load_annotation(self, ann_path: str) -> Dict:
        """Load SA-1B annotation file"""
        if ann_path is None or not os.path.exists(ann_path):
            return None

        try:
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            return ann_data
        except Exception as e:
            print(f"Error loading annotation {ann_path}: {e}")
            return None

    def decode_masks(self, ann_data: Dict, img_size: Tuple[int, int]) -> torch.Tensor:
        """Decode RLE masks from SA-1B annotation"""
        if ann_data is None or 'annotations' not in ann_data:
            return torch.zeros(1, img_size[0], img_size[1])

        masks = []
        for ann in ann_data['annotations']:
            if 'segmentation' in ann:
                # SA-1B uses RLE encoding
                rle = ann['segmentation']
                if isinstance(rle, dict) and 'counts' in rle and 'size' in rle:
                    mask = mask_utils.decode(rle)
                    masks.append(mask)

        if masks:
            # Combine all masks into a single tensor
            combined_mask = np.sum(masks, axis=0)
            combined_mask = (combined_mask > 0).astype(np.float32)

            # Resize mask to target size
            mask_pil = Image.fromarray(combined_mask)
            mask_pil = mask_pil.resize((img_size[1], img_size[0]), Image.NEAREST)
            combined_mask = np.array(mask_pil)

            return torch.from_numpy(combined_mask).unsqueeze(0)
        else:
            return torch.zeros(1, img_size[0], img_size[1])

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            original_size = image.size
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random valid sample
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Apply transform
        image = self.transform(image)

        # Prepare sample
        sample = {
            'image': image,
            'path': img_path,
            'idx': idx,
            'original_size': original_size
        }

        # Load masks if requested
        if self.load_masks and self.annotation_files[idx] is not None:
            ann_data = self.load_annotation(self.annotation_files[idx])
            masks = self.decode_masks(ann_data, (self.img_size, self.img_size))
            sample['mask'] = masks

            # Add metadata if available
            if ann_data:
                sample['image_id'] = ann_data.get('image', {}).get('image_id', '')
                sample['num_annotations'] = len(ann_data.get('annotations', []))

        return sample

    @staticmethod
    def collate_fn(batch):
        """Custom collate function"""
        images = torch.stack([item['image'] for item in batch])
        paths = [item['path'] for item in batch]
        indices = torch.tensor([item['idx'] for item in batch])

        batch_dict = {
            'image': images,
            'path': paths,
            'idx': indices
        }

        # Add masks if available
        if 'mask' in batch[0]:
            masks = torch.stack(
                [item.get('mask', torch.zeros(1, batch[0]['image'].shape[1], batch[0]['image'].shape[2]))
                 for item in batch])
            batch_dict['mask'] = masks

        return