import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple
import random
from huggingface_hub import login
from datasets import load_dataset
from PIL import Image
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class HSGSP_DataLoader:
    """Unified dataset loader for HSGSP project"""

    def __init__(self, config):
        self.config = config

    def _apply_mixup(self, images, labels):
        use_mixup = bool(getattr(self.config, "use_mixup", False))
        mixup_alpha = float(getattr(self.config, "mixup_alpha", 0.0))
        mixup_prob = float(getattr(self.config, "mixup_prob", 0.0))
        if not use_mixup or mixup_alpha <= 0.0 or mixup_prob <= 0.0:
            return images, labels

        if random.random() < mixup_prob:
            batch_size = images.size()[0]
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size).to(images.device)
            images = lam * images + (1 - lam) * images[index]
            labels_a, labels_b = labels, labels[index]
            return images, (labels_a, labels_b, lam)
        return images, labels

    def load_cifar10(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Load and preprocess CIFAR-10 dataset.

        Returns:
            train_loader: augmented training dataset (with stochastic transforms)
            val_loader: validation dataset without augmentation
            test_loader: test dataset without augmentation
            train_eval_loader: clean view of the training set (no augmentation) for evaluation
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.15)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.4)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.4)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
        ] if self.config.data_augmentation else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        train_eval_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.config.validation_split * num_train))

        train_idx, val_idx = indices[split:], indices[:split]

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_eval_subset = Subset(train_eval_dataset, train_idx)

        train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        train_eval_loader = DataLoader(train_eval_subset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        return train_loader, val_loader, test_loader, train_eval_loader

    def load_cifar100(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Load and preprocess CIFAR-100 dataset."""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.15)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.4)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.4)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
        ] if self.config.data_augmentation else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        train_eval_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.config.validation_split * num_train))

        train_idx, val_idx = indices[split:], indices[:split]

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_eval_subset = Subset(train_eval_dataset, train_idx)

        train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        train_eval_loader = DataLoader(train_eval_subset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        return train_loader, val_loader, test_loader, train_eval_loader

    def load_imagenet(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Load and preprocess ImageNet-1k dataset from Hugging Face.

        Returns:
            train_loader: augmented training dataset (with stochastic transforms)
            val_loader: validation dataset without augmentation
            test_loader: test dataset without augmentation (note: ImageNet test split has no labels)
            train_eval_loader: clean view of the training set (no augmentation) for evaluation
        """
        # Login to Hugging Face (use your token)
        login(os.getenv("HF_TOKEN"))

        # Download dataset
        # dataset = load_dataset("ILSVRC/imagenet-1k")
        dataset = load_dataset("zh-plus/tiny-imagenet")

        # Access splits
        train_ds = dataset["train"]
        val_ds = dataset["valid"]
        test_ds = dataset["test"]

        # Define transforms for ImageNet (standard mean/std)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.5) if hasattr(self.config, 'use_cutout') and self.config.use_cutout else lambda x: x,  # Optional Cutout if in config
        ] if self.config.data_augmentation else transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # Wrap Hugging Face datasets with custom transform (since HF datasets return dicts with 'image' and 'label')
        class HFWrapper(torch.utils.data.Dataset):
            def __init__(self, hf_dataset, transform=None):
                self.hf_dataset = hf_dataset
                self.transform = transform

            def __len__(self):
                return len(self.hf_dataset)

            def __getitem__(self, idx):
                item = self.hf_dataset[idx]
                image = item['image'].convert('RGB')  # Ensure RGB
                label = item.get('label', -1)  # Test split may not have labels
                if self.transform:
                    image = self.transform(image)
                return image, label

        # Create wrapped datasets
        train_dataset = HFWrapper(train_ds, transform=transform_train)
        val_dataset = HFWrapper(val_ds, transform=transform_test)
        test_dataset = HFWrapper(test_ds, transform=transform_test)
        train_eval_dataset = HFWrapper(train_ds, transform=transform_test)  # No aug for eval

        # ImageNet has pre-split validation, so no need for validation_split from config
        # But if config.validation_split > 0, we can optionally split train further (though not recommended)
        if self.config.validation_split > 0:
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            split = int(np.floor(self.config.validation_split * num_train))
            train_idx, extra_val_idx = indices[split:], indices[:split]
            train_dataset = Subset(train_dataset, train_idx)
            # Could add to val_dataset, but skipping for simplicity

        # Create DataLoaders (adjust num_workers for large datasets)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        train_eval_loader = DataLoader(train_eval_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, val_loader, test_loader, train_eval_loader