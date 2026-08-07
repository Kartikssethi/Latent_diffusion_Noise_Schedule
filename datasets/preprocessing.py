import os
import torch
from torchvision import transforms
from torch.utils.data import Subset
from PIL import Image

class EnsureRGB:
    """
    Ensures image is converted to 3-channel RGB mode.
    Handles PIL images (converting grayscale/RGBA -> RGB) and Tensors.
    """
    def __call__(self, img):
        if hasattr(img, 'convert'):
            return img.convert('RGB')
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] == 1:
                return img.repeat(3, 1, 1)
            elif img.ndim == 3 and img.shape[0] == 4:
                return img[:3, :, :]
        return img


def get_transforms(image_size=256, crop_size=None, is_train=True):
    """
    Preprocessing pipeline for Latent Diffusion Models (LDM):
    1. RGB 3-Channel conversion (ensures 3 x 256 x 256 shape)
    2. Resizes all datasets to uniform 256x256 resolution
    3. Converts to Tensor ([0.0, 1.0])
    4. Normalizes to [-1.0, 1.0] for LDM input
    """
    transform_list = [
        EnsureRGB(),
        transforms.Resize((image_size, image_size)),
    ]
    
    if is_train:
        transform_list.append(transforms.RandomHorizontalFlip())
        
    if crop_size is not None:
        transform_list.append(transforms.CenterCrop(crop_size))
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    return transforms.Compose(transform_list)


def create_subset(dataset, fraction=0.4, seed=42):
    """
    Subsamples any dataset to retain only `fraction` (default: 40%) of the total data.
    
    Args:
        dataset (torch.utils.data.Dataset): Target PyTorch dataset
        fraction (float): Fraction of data to retain (0.4 = 40% of total data)
        seed (int): Seed for reproducible random sampling across runs
        
    Returns:
        torch.utils.data.Subset: PyTorch subset containing 40% of the dataset
    """
    total_size = len(dataset)
    subset_size = int(total_size * fraction)
    
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator)[:subset_size].tolist()
    
    return Subset(dataset, indices)


def denormalize(tensor):
    """
    Reverses [-1, 1] normalization back to [0, 1] for visual inspection/saving.
    """
    tensor = (tensor * 0.5) + 0.5
    return torch.clamp(tensor, 0.0, 1.0)


def preprocess_and_save_dataset(dataset, save_dir, batch_size=32, num_workers=2, max_batches=None):
    """
    Offline Preprocessing Utility:
    Iterates through dataset (e.g. 40% subset), applies 256x256 RGB [-1, 1] transforms,
    and saves preprocessed PyTorch tensors (.pt) to disk upfront in `save_dir`.

    Args:
        max_batches (int): Optional limit on number of batches to save (useful to avoid disk full).
    """
    os.makedirs(save_dir, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Pre-saving {len(dataset)} preprocessed images to '{save_dir}'...")
    saved_count = 0
    skipped = 0
    for idx, batch in enumerate(dataloader):
        if max_batches is not None and idx >= max_batches:
            break

        save_path = os.path.join(save_dir, f"batch_{idx}.pt")

        # Skip already-saved batches (resume support)
        if os.path.exists(save_path):
            skipped += 1
            continue

        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        try:
            torch.save(images, save_path)
            saved_count += len(images)
        except RuntimeError as e:
            if "iostream error" in str(e) or "enforce fail" in str(e) or "no space" in str(e).lower():
                print(f"\n[WARNING] Disk full at batch {idx}! Stopped early.")
                print(f"Saved {saved_count} images so far in '{save_dir}'. Use these for training.")
                break
            raise

    print(f"Done. Saved: {saved_count} samples | Skipped (already existed): {skipped} | Dir: '{save_dir}'")
    return save_dir


# Default 256x256 RGB transform instance
transform = get_transforms(image_size=256, is_train=True)
