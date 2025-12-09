import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

# Mapping: positive = has ground-truth anomaly masks, negative = classification-only datasets
CLASS_NAMES = ['Brain', 'Liver', 'Retina_RESC', 'Retina_OCT2017', 'Chest', 'Histopathology']
CLASS_INDEX = {
    'Brain': 3, 'Liver': 2, 'Retina_RESC': 1,
    'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3
}


class MedDataset(Dataset):
    """
    Unified dataset loader for medical anomaly detection datasets with the following structure:
    
    <dataset_path>/<class_name>_AD/
    ├── train/
    │   └── good/
    │       └── img/                  → normal training images (no masks)
    ├── valid/
    │   ├── good/
    │   │   └── img/                  → normal validation images
    │   └── Ungood/
    │       ├── img/                  → abnormal validation images
    │       └── anomaly_mask/         → (optional) anomaly masks for seg datasets
    └── test/
        ├── good/
        │   └── img/                  → normal test images
        └── Ungood/
            ├── img/                  → abnormal test images
            └── anomaly_mask/         → (optional) anomaly masks for seg datasets
    """
    def __init__(self,
                 dataset_path='/data/',
                 class_name='Brain',
                 split='test',           # 'train', 'valid', or 'test'
                 resize=240,
                 ):
        assert class_name in CLASS_NAMES, f"class_name: {class_name}, should be in {CLASS_NAMES}"
        assert split in ['train', 'valid', 'test'], f"split must be train/valid/test, got {split}"

        self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.split = split
        self.resize = resize
        self.class_name = class_name
        self.seg_flag = CLASS_INDEX[class_name]  # >0 → has masks, <0 → classification only

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        # Load all images, labels, and masks
        self.images, self.labels, self.masks = self._load_split()

        print(f"[{class_name} - {split}] Loaded {len(self.images)} images "
              f" Here Labels (normal: {self.labels.count(0)}, abnormal: {self.labels.count(1)})")

        # Transforms
        self.transform_img = transforms.Compose([
            transforms.Resize((resize, resize), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((resize, resize), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def _load_split(self):
        images, labels, masks = [], [], []

        if self.split == 'train':
            # Train split: only normal (good) images from train/good/img
            img_dir = os.path.join(self.dataset_path, 'train', 'good')
            img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            images.extend(img_paths)
            labels.extend([0] * len(img_paths))
            masks.extend([None] * len(img_paths))

        print("First 5 images paths found in train/good:")
        for p in img_paths[:5]:
            print(p)
        num_to_print = 5

        print(f"\nDisplaying the first {num_to_print} labels and masks added:")
        for i, (label, mask) in enumerate(zip(labels[:num_to_print], masks[:num_to_print])):
            print(f"Index {i}: Label = {label}, Mask Status = {'Present' if mask is not None else 'None'}")

        else:  # valid or test
            # Load normal (good)
            good_dir = os.path.join(self.dataset_path, self.split, 'good', 'img')
            if os.path.exists(good_dir):
                good_paths = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir)
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                images.extend(good_paths)
                labels.extend([0] * len(good_paths))
                masks.extend([None] * len(good_paths))

            # Load abnormal (Ungood)
            abnorm_img_dir = os.path.join(self.dataset_path, self.split, 'Ungood', 'img')
            abnorm_mask_dir = os.path.join(self.dataset_path, self.split, 'Ungood', 'anomaly_mask')

            if os.path.exists(abnorm_img_dir):
                abnorm_paths = sorted([os.path.join(abnorm_img_dir, f) for f in os.listdir(abnorm_img_dir)
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                images.extend(abnorm_paths)
                labels.extend([1] * len(abnorm_paths))

                if self.seg_flag > 0 and os.path.exists(abnorm_mask_dir):
                    # Assume mask has same filename as image
                    mask_paths = []
                    for img_path in abnorm_paths:
                        mask_name = os.path.basename(img_path)
                        mask_path = os.path.join(abnorm_mask_dir, mask_name)
                        if not os.path.exists(mask_path):
                            raise FileNotFoundError(f"Mask not found: {mask_path}")
                        mask_paths.append(mask_path)
                    masks.extend(mask_paths)
                else:
                    masks.extend([None] * len(abnorm_paths))
            else:
                print(f"Warning: No Ungood images found in {abnorm_img_dir}")

        return images, labels, masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        mask_path = self.masks[idx]

        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform_img(image)

        # Handle mask
        if mask_path is None or self.seg_flag <= 0:
            mask = torch.zeros((1, self.resize, self.resize), dtype=torch.float32)
        else:
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask)  # (1, H, W), values in [0,1]

        return image, label, mask