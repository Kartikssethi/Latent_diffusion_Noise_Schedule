import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import CIFAR10, Flowers102, ImageFolder

from .preprocessing import get_transforms, create_subset, IMAGE_SIZE


class KaggleImageFolderDataset(Dataset):
    """
    Generic Image Dataset for directory of images (e.g. CelebA, LSUN Bedroom, Oxford Flowers).
    Uses fast os.scandir to find image files (.jpg, .jpeg, .png, .webp) in under a second.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.PNG', '.JPEG')
        
        if os.path.exists(root_dir):
            stack = [root_dir]
            while stack:
                curr_dir = stack.pop()
                try:
                    with os.scandir(curr_dir) as entries:
                        for entry in entries:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                            elif entry.name.endswith(valid_extensions):
                                self.image_paths.append(entry.path)
                except Exception:
                    pass
        
        self.image_paths.sort()
        if len(self.image_paths) == 0:
            print(f"Warning: No image files found inside {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Dummy label for generative model / LDM training


class PreprocessedTensorDataset(Dataset):
    """
    Dataset that loads preprocessed tensor batches (.pt files) saved on disk upfront.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')
        ]) if os.path.exists(data_dir) else []
        
        self.samples = []
        for f in self.file_paths:
            try:
                batch = torch.load(f)
                if isinstance(batch, torch.Tensor):
                    for i in range(len(batch)):
                        self.samples.append(batch[i])
            except Exception as e:
                print(f"Error loading {f}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], 0


def find_kaggle_dataset_dir(keywords, candidate_paths=None):
    """
    Fast discovery of dataset directory in Kaggle (/kaggle/input) or local candidate paths.
    Supports both individual named folders and the combined /kaggle/input/datasets/ layout.
    """
    if candidate_paths:
        for path in candidate_paths:
            if path and os.path.exists(path):
                return path

    kaggle_input = "/kaggle/input"
    if not os.path.exists(kaggle_input):
        return None

    # Build a flat list of all directories to search (top-level + /kaggle/input/datasets/)
    search_roots = [kaggle_input]
    combined_datasets_dir = os.path.join(kaggle_input, "datasets")
    if os.path.isdir(combined_datasets_dir):
        search_roots.append(combined_datasets_dir)

    for search_root in search_roots:
        try:
            with os.scandir(search_root) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        entry_lower = entry.name.lower()
                        if any(kw.lower() in entry_lower for kw in keywords):
                            return entry.path
                        # Check 1 level deeper inside each subdirectory
                        try:
                            with os.scandir(entry.path) as sub_entries:
                                for sub in sub_entries:
                                    if sub.is_dir(follow_symlinks=False):
                                        if any(kw.lower() in sub.name.lower() for kw in keywords):
                                            return sub.path
                        except Exception:
                            pass
        except Exception:
            pass
    return None


def get_available_kaggle_inputs():
    """Lists directories present in /kaggle/input for helpful debugging."""
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        try:
            return os.listdir(kaggle_input)
        except Exception:
            pass
    return []


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

    Automatically applies:
    1. 256x256 RGB Transformation & [-1, 1] Normalization (on-the-fly)
    2. Subsampling to only `fraction` (default: 0.4 = 40%) of dataset
    """
    transform = get_transforms(image_size=image_size, is_train=is_train)
    name = dataset_name.lower().replace("-", "").replace("_", "")

    if name == "cifar10":
        cifar_candidate = find_kaggle_dataset_dir(["cifar"], [root_dir, "./data"])
        if cifar_candidate and os.path.exists(cifar_candidate):
            valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')
            has_images = False
            try:
                with os.scandir(cifar_candidate) as entries:
                    has_images = any(e.name.lower().endswith(valid_exts) for e in entries)
            except Exception:
                pass
            
            if has_images:
                dataset = KaggleImageFolderDataset(root_dir=cifar_candidate, transform=transform)
            else:
                download_needed = not os.path.exists(os.path.join(cifar_candidate, 'cifar-10-batches-py'))
                dataset = CIFAR10(root=cifar_candidate, train=is_train, download=download_needed, transform=transform)
        else:
            cifar_root = root_dir or "./data"
            dataset = CIFAR10(root=cifar_root, train=is_train, download=True, transform=transform)

    elif name in ["celeba", "celebafaces"]:
        search_paths = [
            root_dir,
            "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
            "/kaggle/input/celeba-dataset",
            "/kaggle/input/celeba",
            "./data/celeba"
        ]
        valid_path = find_kaggle_dataset_dir(["celeba"], search_paths)
        if not valid_path:
            avail = get_available_kaggle_inputs()
            raise FileNotFoundError(
                f"CelebA dataset directory not found. Available folders in /kaggle/input: {avail}"
            )
        dataset = KaggleImageFolderDataset(root_dir=valid_path, transform=transform)

    elif name in ["flowers", "oxfordflowers", "oxfordflowers102"]:
        search_paths = [
            root_dir,
            "/kaggle/input/the-oxford-flowers-102-dataset",
            "/kaggle/input/oxford-102-flower-pytorch",
            "/kaggle/input/flowers-102",
            "./data/flowers"
        ]
        valid_path = find_kaggle_dataset_dir(["flower", "oxford"], search_paths)
        if valid_path:
            dataset = KaggleImageFolderDataset(root_dir=valid_path, transform=transform)
        else:
            dataset = Flowers102(root="./data", split="train" if is_train else "test", download=True, transform=transform)

    elif name in ["lsun", "lsunbedroom"]:
        search_paths = [
            root_dir,
            "/kaggle/input/lsun-bedroom-scene-20-sample",
            "/kaggle/input/lsun-bedroom",
            "/kaggle/input/lsun",
            "./data/lsun"
        ]
        valid_path = find_kaggle_dataset_dir(["lsun", "bedroom"], search_paths)
        if not valid_path:
            avail = get_available_kaggle_inputs()
            raise FileNotFoundError(
                f"LSUN dataset directory not found. Available folders in /kaggle/input: {avail}"
            )
        dataset = KaggleImageFolderDataset(root_dir=valid_path, transform=transform)

    else:
        valid_path = root_dir if (root_dir and os.path.exists(root_dir)) else find_kaggle_dataset_dir([name])
        if valid_path:
            dataset = KaggleImageFolderDataset(root_dir=valid_path, transform=transform)
        else:
            avail = get_available_kaggle_inputs()
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available folders in /kaggle/input: {avail}")

    if fraction < 1.0:
        dataset = create_subset(dataset, fraction=fraction)

    return dataset
