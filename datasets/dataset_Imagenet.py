import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import random
from typing import Optional, Tuple


class ImageNetDataset(Dataset):
    """
    ImageNet dataset loader for universal adversarial perturbation generation
    """

    def __init__(
            self,
            root_dir: str,
            img_size: int = 512,
            num_samples: Optional[int] = None,
            split: str = 'train',
            seed: int = 42
    ):
        """
        Args:
            root_dir: Path to ImageNet dataset root directory
            img_size: Target image size for resizing
            num_samples: Number of samples to use (None for all)
            split: 'train' or 'val'
            seed: Random seed for sample selection
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split

        # Set up paths
        if split == 'train':
            self.data_dir = os.path.join(root_dir, 'train')
        else:
            self.data_dir = os.path.join(root_dir, 'val')

        # Get all image paths
        self.image_paths = []
        self.labels = []

        if os.path.exists(self.data_dir):
            # For ImageNet structure with class folders
            class_folders = sorted(os.listdir(self.data_dir))
            for class_idx, class_folder in enumerate(class_folders):
                class_path = os.path.join(self.data_dir, class_folder)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
                            self.image_paths.append(os.path.join(class_path, img_name))
                            self.labels.append(class_idx)
        else:
            # Alternative: flat directory structure
            self.data_dir = root_dir
            for img_name in os.listdir(self.data_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
                    self.image_paths.append(os.path.join(self.data_dir, img_name))
                    self.labels.append(0)  # Dummy label

        # Sample if requested
        if num_samples is not None and num_samples < len(self.image_paths):
            random.seed(seed)
            indices = random.sample(range(len(self.image_paths)), num_samples)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        # Additional augmentation transforms
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])

        print(f"Loaded {len(self.image_paths)} images from ImageNet {split} set")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Apply transforms
        image = self.transform(image)

        # Ensure image has 3 channels
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3, :, :]

        # Get label
        label = self.labels[idx]

        # Create sample dictionary
        sample = {
            'image': image,
            'label': label,
            'path': img_path,
            'idx': idx
        }

        return sample

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching"""
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        paths = [item['path'] for item in batch]
        indices = torch.tensor([item['idx'] for item in batch])

        return {
            'image': images,
            'label': labels,
            'path': paths,
            'idx': indices
        }


class ImageNetSubset(Dataset):
    """
    A subset of ImageNet for faster experimentation
    """

    def __init__(
            self,
            root_dir: str,
            img_size: int = 512,
            classes: list = None,
            samples_per_class: int = 100
    ):
        """
        Args:
            root_dir: Path to ImageNet dataset
            img_size: Target image size
            classes: List of class indices to use (None for random selection)
            samples_per_class: Number of samples per class
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.data_dir = os.path.join(root_dir, 'train')

        # Get class folders
        all_classes = sorted(os.listdir(self.data_dir))

        # Select classes
        if classes is None:
            # Randomly select 100 classes
            classes = random.sample(range(len(all_classes)), min(100, len(all_classes)))

        self.image_paths = []
        self.labels = []

        for class_idx in classes:
            class_folder = all_classes[class_idx]
            class_path = os.path.join(self.data_dir, class_folder)

            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG'))]

                # Sample images from this class
                num_samples = min(samples_per_class, len(images))
                sampled_images = random.sample(images, num_samples)

                for img_name in sampled_images:
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(class_idx)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        print(f"Created ImageNet subset with {len(self.image_paths)} images from {len(classes)} classes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except:
            # Return random sample if loading fails
            return self.__getitem__(random.randint(0, len(self) - 1))

        return {
            'image': image,
            'label': self.labels[idx],
            'path': img_path
        }