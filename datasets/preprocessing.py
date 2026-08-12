"""
Data Preprocessing Pipeline for Latent Diffusion Noise Schedule Project.

This module handles the complete data preprocessing workflow:
1. Load 4 datasets: LSUN Bedroom, CelebA, Oxford Flowers, CIFAR-10
2. Randomly select 40% from each dataset (reproducible via fixed seed)
3. Resize all images to 128x128 RGB
4. Apply data augmentation (random horizontal flip)
5. Normalize pixel values to [-1, 1]
6. Combine into a single training dataset with source labels
7. Verify dataset integrity
8. Save preprocessed tensors to disk
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
from PIL import Image, ImageFile

# Allow loading of truncated images to prevent crashes on corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 128
SAMPLE_FRACTION = 0.4
RANDOM_SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 4  # Adjust based on hardware
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# Source label mapping
SOURCE_LABELS = {
    "lsun_bedroom": 0,
    "celeba": 1,
    "oxford_flowers": 2,
    "cifar10": 3,
}

SOURCE_NAMES = {v: k for k, v in SOURCE_LABELS.items()}

# Resolve project root (two levels up from this file)
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_data"

# Dataset paths within data/
LSUN_BEDROOM_DIR = DATA_DIR / "bedroom"
CELEBA_DIR = DATA_DIR / "celebA"
FLOWERS_DIR = DATA_DIR / "102flowers"
CIFAR10_DIR = DATA_DIR / "cifar10"

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ============================= Helper Classes ==============================


class EnsureRGB:
    """
    Ensures image is converted to 3-channel RGB mode.
    Handles PIL images (converting grayscale/RGBA -> RGB) and Tensors.
    """

    def __call__(self, img):
        if hasattr(img, "convert"):
            return img.convert("RGB")
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] == 1:
                return img.repeat(3, 1, 1)
            elif img.ndim == 3 and img.shape[0] == 4:
                return img[:3, :, :]
        return img


class ImageFolderDataset(Dataset):
    """
    Generic image dataset that recursively scans a directory for image files.
    Uses fast os.scandir for discovery. Returns (image, source_label).
    """

    def __init__(self, root_dir, source_label, transform=None):
        self.root_dir = str(root_dir)
        self.source_label = source_label
        self.transform = transform
        self.image_paths = []

        if os.path.exists(self.root_dir):
            self._scan_directory(self.root_dir)

        self.image_paths.sort()
        if len(self.image_paths) == 0:
            logger.warning(f"No image files found in {self.root_dir}")

    def _scan_directory(self, directory):
        """Recursively scan directory for image files using os.scandir (fast)."""
        stack = [directory]
        while stack:
            curr_dir = stack.pop()
            try:
                with os.scandir(curr_dir) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in VALID_IMAGE_EXTENSIONS:
                                self.image_paths.append(entry.path)
            except PermissionError:
                pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path)
            # Force load to catch truncated images early
            image.load()
            image = image.convert("RGB")
        except Exception as e:
            logger.warning(f"Corrupted image skipped: {img_path} ({e})")
            # Return a black placeholder image so DataLoader doesn't crash
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, self.source_label


class CombinedDatasetWithLabels(Dataset):
    """
    Wrapper around ConcatDataset that ensures source labels are preserved.
    Each item returns (image_tensor, source_label_int).
    """

    def __init__(self, concat_dataset):
        self.dataset = concat_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# ============================= Transform Pipelines =========================


def get_transforms(image_size=IMAGE_SIZE, is_train=True):
    """
    Preprocessing pipeline for LDM training:
    1. Convert to 3-channel RGB
    2. Resize to image_size x image_size (default 128x128)
    3. Random horizontal flip (training only)
    4. Convert to tensor [0, 1]
    5. Normalize to [-1, 1] using mean=0.5, std=0.5 per channel
    """
    transform_list = [
        EnsureRGB(),
        transforms.Resize((image_size, image_size)),
    ]

    if is_train:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )

    return transforms.Compose(transform_list)


# ============================= Subset Sampling =============================


def create_subset(dataset, fraction=SAMPLE_FRACTION, seed=RANDOM_SEED):
    """
    Randomly sample `fraction` of the dataset using a fixed seed.

    This does NOT simply take the first N items — it generates a random
    permutation of all indices and selects the first `fraction * len(dataset)`
    indices from the permutation, ensuring reproducibility.

    Args:
        dataset: PyTorch Dataset to sample from
        fraction: Fraction of data to retain (0.4 = 40%)
        seed: Random seed for reproducibility

    Returns:
        torch.utils.data.Subset containing the randomly selected items
    """
    total_size = len(dataset)
    subset_size = int(total_size * fraction)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator)[:subset_size].tolist()

    logger.info(f"  Sampled {subset_size}/{total_size} images ({fraction*100:.0f}%)")
    return Subset(dataset, indices)


# ============================= Denormalization =============================


def denormalize(tensor):
    """
    Reverses [-1, 1] normalization back to [0, 1] for visual inspection/saving.
    """
    tensor = (tensor * 0.5) + 0.5
    return torch.clamp(tensor, 0.0, 1.0)


# ============================= Dataset Loading =============================


def load_all_datasets(image_size=IMAGE_SIZE, fraction=SAMPLE_FRACTION, seed=RANDOM_SEED):
    """
    Load all 4 datasets, apply transforms, and randomly sample `fraction` from each.

    Returns:
        combined_dataset: ConcatDataset of all sampled subsets
        dataset_info: dict with per-dataset statistics
    """
    transform = get_transforms(image_size=image_size, is_train=True)
    dataset_info = {}

    # ---- 1. LSUN Bedroom ----
    logger.info("Loading LSUN Bedroom dataset...")
    lsun_dataset = ImageFolderDataset(
        root_dir=LSUN_BEDROOM_DIR,
        source_label=SOURCE_LABELS["lsun_bedroom"],
        transform=transform,
    )
    lsun_full_count = len(lsun_dataset)
    lsun_subset = create_subset(lsun_dataset, fraction=fraction, seed=seed)
    dataset_info["lsun_bedroom"] = {
        "full_count": lsun_full_count,
        "subset_count": len(lsun_subset),
        "source_label": SOURCE_LABELS["lsun_bedroom"],
    }

    # ---- 2. CelebA ----
    logger.info("Loading CelebA dataset...")
    celeba_dataset = ImageFolderDataset(
        root_dir=CELEBA_DIR,
        source_label=SOURCE_LABELS["celeba"],
        transform=transform,
    )
    celeba_full_count = len(celeba_dataset)
    celeba_subset = create_subset(celeba_dataset, fraction=fraction, seed=seed)
    dataset_info["celeba"] = {
        "full_count": celeba_full_count,
        "subset_count": len(celeba_subset),
        "source_label": SOURCE_LABELS["celeba"],
    }

    # ---- 3. Oxford Flowers 102 ----
    logger.info("Loading Oxford Flowers dataset...")
    flowers_dataset = ImageFolderDataset(
        root_dir=FLOWERS_DIR,
        source_label=SOURCE_LABELS["oxford_flowers"],
        transform=transform,
    )
    flowers_full_count = len(flowers_dataset)
    flowers_subset = create_subset(flowers_dataset, fraction=fraction, seed=seed)
    dataset_info["oxford_flowers"] = {
        "full_count": flowers_full_count,
        "subset_count": len(flowers_subset),
        "source_label": SOURCE_LABELS["oxford_flowers"],
    }

    # ---- 4. CIFAR-10 ----
    logger.info("Loading CIFAR-10 dataset...")
    # CIFAR-10 has train/ and test/ subdirectories with class subfolders
    # We load both train and test for maximum data
    cifar_train = ImageFolderDataset(
        root_dir=CIFAR10_DIR / "train",
        source_label=SOURCE_LABELS["cifar10"],
        transform=transform,
    )
    cifar_test = ImageFolderDataset(
        root_dir=CIFAR10_DIR / "test",
        source_label=SOURCE_LABELS["cifar10"],
        transform=transform,
    )
    # Combine train + test into single CIFAR-10 dataset
    cifar_combined = ConcatDataset([cifar_train, cifar_test])
    cifar_full_count = len(cifar_combined)
    cifar_subset = create_subset(cifar_combined, fraction=fraction, seed=seed)
    dataset_info["cifar10"] = {
        "full_count": cifar_full_count,
        "subset_count": len(cifar_subset),
        "source_label": SOURCE_LABELS["cifar10"],
    }

    # ---- Combine all subsets ----
    combined_dataset = ConcatDataset([lsun_subset, celeba_subset, flowers_subset, cifar_subset])
    total_count = len(combined_dataset)

    logger.info(f"\n{'='*60}")
    logger.info("DATASET SUMMARY")
    logger.info(f"{'='*60}")
    for name, info in dataset_info.items():
        logger.info(
            f"  {name:20s}: {info['subset_count']:>7,} / {info['full_count']:>7,} "
            f"(label={info['source_label']})"
        )
    logger.info(f"  {'TOTAL':20s}: {total_count:>7,}")
    logger.info(f"{'='*60}\n")

    return CombinedDatasetWithLabels(combined_dataset), dataset_info


# ============================= DataLoader ==================================


def create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS):
    """
    Create a PyTorch DataLoader for the combined dataset.

    Args:
        dataset: Combined preprocessed dataset
        batch_size: Batch size (default 16)
        shuffle: Whether to shuffle data (True for training)
        num_workers: Number of parallel data-loading workers

    Returns:
        DataLoader instance
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    logger.info(
        f"DataLoader created: batch_size={batch_size}, shuffle={shuffle}, "
        f"num_workers={num_workers}, total_batches={len(dataloader)}"
    )
    return dataloader


# ============================= Verification ================================


def verify_dataset(dataset, dataset_info, num_check=100):
    """
    Verify dataset integrity:
    - Check total image count
    - Verify image shape = [3, 128, 128]
    - Verify pixel range ≈ [-1, 1]
    - Check source label distribution
    - Detect corrupted images (all-black placeholders)

    Args:
        dataset: The combined dataset to verify
        dataset_info: Dict with per-dataset statistics
        num_check: Number of random samples to inspect

    Returns:
        dict with verification results
    """
    logger.info("Verifying dataset integrity...")
    total = len(dataset)
    results = {
        "total_images": total,
        "per_dataset": {},
        "shape_check_passed": True,
        "pixel_range_check_passed": True,
        "corrupted_count": 0,
        "source_label_distribution": Counter(),
    }

    # Check counts match expected
    expected_total = sum(info["subset_count"] for info in dataset_info.values())
    assert total == expected_total, (
        f"Total mismatch: dataset has {total} but expected {expected_total}"
    )
    results["per_dataset"] = {
        name: info["subset_count"] for name, info in dataset_info.items()
    }

    # Sample random indices for detailed checks
    check_indices = random.sample(range(total), min(num_check, total))
    pixel_mins = []
    pixel_maxs = []

    for idx in check_indices:
        img, label = dataset[idx]

        # Shape check
        if img.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
            logger.error(f"Shape mismatch at idx {idx}: {img.shape}")
            results["shape_check_passed"] = False

        # Pixel range check
        pmin, pmax = img.min().item(), img.max().item()
        pixel_mins.append(pmin)
        pixel_maxs.append(pmax)

        if pmin < -1.1 or pmax > 1.1:
            logger.warning(f"Pixel range issue at idx {idx}: [{pmin:.4f}, {pmax:.4f}]")
            results["pixel_range_check_passed"] = False

        # Corruption check (all-zero tensor = black placeholder)
        if img.abs().sum().item() < 1e-6:
            results["corrupted_count"] += 1

        results["source_label_distribution"][label] += 1

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("VERIFICATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Total images:          {total:,}")
    for name, count in results["per_dataset"].items():
        logger.info(f"  {name:22s}: {count:>7,}")
    logger.info(f"  Image shape check:     {'PASS' if results['shape_check_passed'] else 'FAIL'}")
    logger.info(f"  Pixel range check:     {'PASS' if results['pixel_range_check_passed'] else 'FAIL'}")
    if pixel_mins:
        logger.info(f"  Pixel min range:       [{min(pixel_mins):.4f}, {max(pixel_mins):.4f}]")
        logger.info(f"  Pixel max range:       [{min(pixel_maxs):.4f}, {max(pixel_maxs):.4f}]")
    logger.info(f"  Corrupted images:      {results['corrupted_count']} (out of {len(check_indices)} checked)")
    logger.info(f"  Source label dist:     {dict(results['source_label_distribution'])}")
    logger.info(f"{'='*60}\n")

    return results


# ============================= Save Preprocessed ==========================


def save_preprocessed_dataset(
    dataset,
    save_dir=None,
    batch_size=64,
    num_workers=NUM_WORKERS,
    max_batches=None,
):
    """
    Save the entire preprocessed dataset as PyTorch tensor batches (.pt) to disk.
    Each .pt file contains a dict: {'images': tensor, 'labels': tensor}.

    Supports resume — skips already-saved batch files.

    Args:
        dataset: Combined preprocessed dataset
        save_dir: Output directory (default: preprocessed_data/)
        batch_size: Number of images per saved .pt file
        num_workers: Workers for data loading
        max_batches: Optional limit on batches to save

    Returns:
        Path to the save directory
    """
    if save_dir is None:
        save_dir = str(PREPROCESSED_DIR)

    os.makedirs(save_dir, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    total_batches = len(dataloader)
    logger.info(f"Saving {len(dataset)} preprocessed images to '{save_dir}'...")
    logger.info(f"  Batch size: {batch_size}, Total batches: {total_batches}")

    saved_count = 0
    skipped_count = 0
    start_time = time.time()

    for idx, (images, labels) in enumerate(dataloader):
        if max_batches is not None and idx >= max_batches:
            break

        save_path = os.path.join(save_dir, f"batch_{idx:06d}.pt")

        # Resume support: skip already-saved batches
        if os.path.exists(save_path):
            skipped_count += 1
            continue

        try:
            torch.save({"images": images, "labels": labels}, save_path)
            saved_count += len(images)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "no space" in err_msg or "iostream" in err_msg or "enforce fail" in err_msg:
                logger.error(f"Disk full at batch {idx}! Stopped early.")
                logger.error(f"Saved {saved_count} images so far in '{save_dir}'.")
                break
            raise

        # Progress logging every 100 batches
        if (idx + 1) % 100 == 0 or idx == total_batches - 1:
            elapsed = time.time() - start_time
            rate = saved_count / max(elapsed, 1e-6)
            logger.info(
                f"  Batch {idx+1}/{total_batches} | "
                f"Saved: {saved_count:,} | "
                f"Rate: {rate:.0f} img/s | "
                f"Elapsed: {elapsed:.1f}s"
            )

    elapsed = time.time() - start_time

    # Save metadata
    metadata = {
        "total_images": saved_count + (skipped_count * batch_size),
        "batch_size": batch_size,
        "total_batches": total_batches,
        "image_size": IMAGE_SIZE,
        "channels": 3,
        "normalization": {"mean": NORMALIZE_MEAN, "std": NORMALIZE_STD},
        "sample_fraction": SAMPLE_FRACTION,
        "random_seed": RANDOM_SEED,
        "source_labels": SOURCE_LABELS,
    }
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nPreprocessing complete!")
    logger.info(f"  Saved: {saved_count:,} images")
    logger.info(f"  Skipped (already existed): {skipped_count} batches")
    logger.info(f"  Output directory: {save_dir}")
    logger.info(f"  Metadata saved to: {metadata_path}")
    logger.info(f"  Total time: {elapsed:.1f}s")

    return save_dir


# ============================= Load Preprocessed ==========================


class PreprocessedTensorDataset(Dataset):
    """
    Dataset that loads preprocessed tensor batches (.pt files) from disk.
    Each .pt file contains {'images': tensor, 'labels': tensor}.
    Also supports legacy format where .pt files contain just image tensors.
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = str(PREPROCESSED_DIR)

        self.data_dir = data_dir
        self.samples = []  # list of (image_tensor, label)
        self.file_paths = []

        if os.path.exists(data_dir):
            self.file_paths = sorted(
                [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]
            )

        logger.info(f"Loading {len(self.file_paths)} batch files from '{data_dir}'...")
        for fpath in self.file_paths:
            try:
                data = torch.load(fpath, weights_only=True)
                if isinstance(data, dict):
                    images = data["images"]
                    labels = data.get("labels", torch.zeros(len(images), dtype=torch.long))
                    for i in range(len(images)):
                        self.samples.append((images[i], labels[i].item()))
                elif isinstance(data, torch.Tensor):
                    # Legacy format: just tensors, no labels
                    for i in range(len(data)):
                        self.samples.append((data[i], 0))
            except Exception as e:
                logger.error(f"Error loading {fpath}: {e}")

        logger.info(f"Loaded {len(self.samples)} preprocessed images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================= Main Entry Point ============================


def run_preprocessing():
    """
    Full preprocessing pipeline:
    1. Load all 4 datasets from data/
    2. Randomly sample 40% from each
    3. Combine with source labels
    4. Verify dataset integrity
    5. Create DataLoader (batch_size=16)
    6. Save preprocessed tensors to preprocessed_data/
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  Image size:      {IMAGE_SIZE}x{IMAGE_SIZE}")
    logger.info(f"  Sample fraction: {SAMPLE_FRACTION * 100:.0f}%")
    logger.info(f"  Random seed:     {RANDOM_SEED}")
    logger.info(f"  Batch size:      {BATCH_SIZE}")
    logger.info(f"  Normalization:   mean={NORMALIZE_MEAN}, std={NORMALIZE_STD}")
    logger.info(f"  Data directory:  {DATA_DIR}")
    logger.info(f"  Output directory: {PREPROCESSED_DIR}")
    logger.info("")

    # Step 1 & 2: Load datasets and sample 40%
    combined_dataset, dataset_info = load_all_datasets(
        image_size=IMAGE_SIZE, fraction=SAMPLE_FRACTION, seed=RANDOM_SEED
    )

    # Step 3: Verify dataset integrity
    verification = verify_dataset(combined_dataset, dataset_info, num_check=200)

    # Step 4: Create DataLoader for training
    dataloader = create_dataloader(
        combined_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    # Quick sanity check on first batch
    logger.info("Fetching first batch for sanity check...")
    first_batch_images, first_batch_labels = next(iter(dataloader))
    logger.info(f"  First batch images shape: {first_batch_images.shape}")
    logger.info(f"  First batch labels:       {first_batch_labels.tolist()}")
    logger.info(f"  Pixel range:              [{first_batch_images.min():.4f}, {first_batch_images.max():.4f}]")

    # Step 5: Save preprocessed data to disk
    save_preprocessed_dataset(
        combined_dataset,
        save_dir=str(PREPROCESSED_DIR),
        batch_size=64,
        num_workers=NUM_WORKERS,
    )

    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return combined_dataset, dataloader, dataset_info


# Default 128x128 RGB transform instance (for use by other modules)
transform = get_transforms(image_size=IMAGE_SIZE, is_train=True)


if __name__ == "__main__":
    run_preprocessing()
