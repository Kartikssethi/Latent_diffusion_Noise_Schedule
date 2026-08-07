import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import CIFAR10, Flowers102, ImageFolder

from .preprocessing import get_transforms, create_subset


class KaggleImageFolderDataset(Dataset):
    """
    Generic Image Dataset for directory of images (e.g. CelebA, LSUN Bedroom, Oxford Flowers).
    Finds all image files (.jpg, .jpeg, .png, .webp) in root or any subfolder.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Support single folder of images or nested subfolders
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.PNG', '.JPEG')
        self.image_paths = []
        
        if os.path.exists(root_dir):
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if f.endswith(valid_extensions):
                        self.image_paths.append(os.path.join(root, f))
        
        self.image_paths.sort()
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Dummy label for generative model / LDM training


def load_cifar10(root="./data", image_size=256, is_train=True, fraction=0.4):
    """Loads CIFAR-10 with LDM 256x256 preprocessing."""
    transform = get_transforms(image_size=image_size, is_train=is_train)
    dataset = CIFAR10(root=root, train=is_train, download=True, transform=transform)
    if fraction < 1.0:
        dataset = create_subset(dataset, fraction=fraction)
    return dataset


def load_dataset(dataset_name, root_dir=None, image_size=256, is_train=True, fraction=0.4):
    """
    Unified loader for Kaggle & local datasets:
    - 'cifar10'
    - 'celeba'
    - 'flowers' (Oxford Flowers 102)
    - 'lsun' (LSUN Bedroom)
    """
    transform = get_transforms(image_size=image_size, is_train=is_train)
    name = dataset_name.lower().replace("-", "").replace("_", "")

    # Kaggle default input directory lookup if root_dir not explicitly provided
    kaggle_input = "/kaggle/input"
    
    if name == "cifar10":
        cifar_root = root_dir or "./data"
        dataset = CIFAR10(root=cifar_root, train=is_train, download=True, transform=transform)
        
    elif name in ["celeba", "celebafaces"]:
        search_paths = [
            root_dir,
            "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
            "/kaggle/input/celeba-dataset",
            "/kaggle/input/celeba/img_align_celeba",
            "./data/celeba"
        ]
        valid_path = next((p for p in search_paths if p and os.path.exists(p)), None)
        if not valid_path:
            raise FileNotFoundError(f"CelebA dataset directory not found in candidate paths: {search_paths}")
        dataset = KaggleImageFolderDataset(root_dir=valid_path, transform=transform)

    elif name in ["flowers", "oxfordflowers", "oxfordflowers102"]:
        search_paths = [
            root_dir,
            "/kaggle/input/the-oxford-flowers-102-dataset",
            "/kaggle/input/oxford-102-flower-pytorch",
            "/kaggle/input/flowers-102",
            "./data/flowers"
        ]
        valid_path = next((p for p in search_paths if p and os.path.exists(p)), None)
        if valid_path:
            dataset = KaggleImageFolderDataset(root_dir=valid_path, transform=transform)
        else:
            # Fallback to torchvision Flowers102 download
            dataset = Flowers102(root="./data", split="train" if is_train else "test", download=True, transform=transform)

    elif name in ["lsun", "lsunbedroom"]:
        search_paths = [
            root_dir,
            "/kaggle/input/lsun-bedroom-scene-20-sample",
            "/kaggle/input/lsun-bedroom",
            "./data/lsun"
        ]
        valid_path = next((p for p in search_paths if p and os.path.exists(p)), None)
        if not valid_path:
            raise FileNotFoundError(f"LSUN dataset directory not found in candidate paths: {search_paths}")
        dataset = KaggleImageFolderDataset(root_dir=valid_path, transform=transform)

    else:
        # Fallback generic directory loader
        if root_dir and os.path.exists(root_dir):
            dataset = KaggleImageFolderDataset(root_dir=root_dir, transform=transform)
        else:
            raise ValueError(f"Unknown dataset '{dataset_name}' and invalid root_dir: {root_dir}")

    if fraction < 1.0:
        dataset = create_subset(dataset, fraction=fraction)
        
    return dataset
