"""
Training Script for the Adaptive Noise Scheduler.

This script trains ONLY the AdaptiveNoiseScheduler module — learning optimal
beta_t values for the forward diffusion process. The UNet denoiser and VAE
are NOT trained here.

Training objective:
    - VLB reconstruction proxy loss (signal preservation quality)
    - SNR uniformity loss (smooth, evenly-spaced log-SNR curve)
    - Monotonicity penalty (alpha_bar must strictly decrease)
    - Boundary constraints (alpha_bar[0] ≈ 1, alpha_bar[T] ≈ 0)

Usage:
    # Quick test on first 100 batches (default):
    python training/train.py

    # Full training on all ~3500 batches:
    python training/train.py --max_batches 0 --epochs 50

    # Resume from checkpoint:
    python training/train.py --resume outputs/checkpoints/adaptive_scheduler_latest.pt
"""

import os
import sys
import argparse
import time
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import models/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.adaptive_scheduler import AdaptiveNoiseScheduler

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Optional: matplotlib for schedule plots
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset: Load preprocessed .pt batch files from disk
# ---------------------------------------------------------------------------

class PreprocessedBatchDataset(Dataset):
    """
    Fast file-level batch dataset for preprocessed .pt files.
    
    Loads preprocessed .pt batch files sequentially/shuffled at the file level 
    rather than seeking individual images across 3,500 files. Reduces disk reads
    from 224,000 down to 3,500 per epoch, accelerating speed by ~100x.
    """

    def __init__(self, data_dir, max_batches=100):
        self.data_dir = data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        batch_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pt")
        ])

        if len(batch_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")

        if max_batches and max_batches > 0:
            batch_files = batch_files[:max_batches]

        self.batch_files = batch_files
        print(f"Fast Dataset Ready: {len(self.batch_files)} batch files found.")

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        fpath = self.batch_files[idx]
        data = torch.load(fpath, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            return data["images"]
        return data


# ---------------------------------------------------------------------------
# Schedule Visualization
# ---------------------------------------------------------------------------

def plot_schedule_comparison(scheduler, epoch, save_dir):
    """
    Plot the learned adaptive schedule vs linear & cosine baselines.

    Generates two subplots:
        1. beta_t vs timestep
        2. alpha_bar_t vs timestep

    Saves to: {save_dir}/schedule_epoch_{epoch}.png
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available — skipping schedule plot.")
        return

    os.makedirs(save_dir, exist_ok=True)
    schedules = scheduler.get_schedule_comparison()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    # --- Plot 1: Beta schedule ---
    ax1 = axes[0]
    T = len(schedules["linear"]["betas"])
    timesteps = range(T)

    ax1.plot(timesteps, schedules["linear"]["betas"].numpy(),
             label="Linear", alpha=0.7, linewidth=1.5, color="#1f77b4")
    ax1.plot(timesteps, schedules["cosine"]["betas"].numpy(),
             label="Cosine", alpha=0.7, linewidth=1.5, color="#2ca02c")
    if "adaptive" in schedules:
        ax1.plot(timesteps, schedules["adaptive"]["betas"].numpy(),
                 label="Adaptive (Learned)", linewidth=2.0, color="#d62728")

    ax1.set_xlabel("Timestep t", fontsize=12)
    ax1.set_ylabel("β_t", fontsize=12)
    ax1.set_title(f"Noise Schedule β_t — Epoch {epoch}", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Alpha-bar schedule ---
    ax2 = axes[1]

    ax2.plot(timesteps, schedules["linear"]["alpha_bar"].numpy(),
             label="Linear", alpha=0.7, linewidth=1.5, color="#1f77b4")
    ax2.plot(timesteps, schedules["cosine"]["alpha_bar"].numpy(),
             label="Cosine", alpha=0.7, linewidth=1.5, color="#2ca02c")
    if "adaptive" in schedules:
        ax2.plot(timesteps, schedules["adaptive"]["alpha_bar"].numpy(),
                 label="Adaptive (Learned)", linewidth=2.0, color="#d62728")

    ax2.set_xlabel("Timestep t", fontsize=12)
    ax2.set_ylabel("ᾱ_t", fontsize=12)
    ax2.set_title(f"Cumulative Signal Retention ᾱ_t — Epoch {epoch}", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"schedule_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(loss_history, save_dir):
    """
    Plot training loss curves over epochs.

    Saves to: {save_dir}/loss_curves.png
    """
    if not HAS_MATPLOTLIB or len(loss_history) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(loss_history) + 1)
    keys = [
        ("total_loss", "Total Loss", "#d62728"),
        ("recon_loss", "Reconstruction Loss", "#1f77b4"),
        ("uniformity_loss", "SNR Uniformity Loss", "#2ca02c"),
        ("monotonicity_loss", "Monotonicity Penalty", "#ff7f0e"),
    ]

    for ax, (key, title, color) in zip(axes.flat, keys):
        values = [h[key] for h in loss_history if key in h]
        if values:
            ax.plot(epochs[:len(values)], values, color=color, linewidth=1.5)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_adaptive_scheduler(args):
    """
    Main training function for the Adaptive Noise Scheduler.

    Only the ScheduleNet MLP parameters are optimized. The loss encourages
    the learned schedule to:
        - Preserve reconstruction quality (VLB proxy)
        - Produce uniformly-spaced log-SNR values
        - Maintain strict monotonicity of alpha_bar
        - Satisfy boundary conditions (alpha_bar: 1 → 0)
    """
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Create output directories
    # ------------------------------------------------------------------
    checkpoint_dir = os.path.join(_PROJECT_ROOT, "outputs", "checkpoints")
    plots_dir = os.path.join(_PROJECT_ROOT, "outputs", "plots")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_PROJECT_ROOT, data_dir)

    dataset = PreprocessedBatchDataset(
        data_dir=data_dir,
        max_batches=args.max_batches if args.max_batches > 0 else None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # 3. Create Adaptive Noise Scheduler
    # ------------------------------------------------------------------
    scheduler = AdaptiveNoiseScheduler(
        timesteps=args.timesteps,
        schedule_type="adaptive",
        beta_min=1e-4,
        beta_max=0.02,
        hidden_dim=256,
    ).to(device)

    trainable_params = sum(p.numel() for p in scheduler.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # 4. Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        scheduler.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
    )

    # Cosine annealing LR schedule
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # ------------------------------------------------------------------
    # 5. Resume from checkpoint (if requested)
    # ------------------------------------------------------------------
    start_epoch = 0
    best_loss = float("inf")
    loss_history = []

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        loss_history = checkpoint.get("loss_history", [])

    # ------------------------------------------------------------------
    # 6. Mixed Precision setup
    # ------------------------------------------------------------------
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    print(f"\nTraining on {device} | {args.epochs} epochs | batch_size={args.batch_size} | lr={args.lr}")
    print(f"{'Epoch':<8} {'Loss':<14} {'Recon':<14} {'Best':<14} {'Elapsed':>10}")
    print("-" * 64)

    total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        scheduler.train()

        epoch_losses = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "uniformity_loss": 0.0,
            "monotonicity_loss": 0.0,
            "boundary_loss": 0.0,
        }
        num_batches = 0

        for batch_idx, images in enumerate(dataloader):
            if images.dim() == 5:
                images = images.squeeze(0)  # [1, 64, 3, 128, 128] -> [64, 3, 128, 128]
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, loss_dict = scheduler.compute_schedule_loss(images)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(scheduler.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Accumulate epoch losses
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            num_batches += 1

        # --- End of epoch ---
        lr_scheduler.step()

        # Average epoch losses
        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        loss_history.append(avg_losses)

        # Elapsed time
        elapsed_total = time.time() - total_start
        elapsed_m, elapsed_s = divmod(int(elapsed_total), 60)
        elapsed_h, elapsed_m = divmod(elapsed_m, 60)
        elapsed_str = f"{elapsed_h}:{elapsed_m:02d}:{elapsed_s:02d}" if elapsed_h else f"{elapsed_m:02d}:{elapsed_s:02d}"

        # One clean line per epoch
        is_best = avg_losses['total_loss'] < best_loss
        best_marker = " *" if is_best else ""
        print(
            f"{epoch+1:<8} {avg_losses['total_loss']:<14.6f} {avg_losses['recon_loss']:<14.6f} {best_loss:<14.6f} {elapsed_str:>10}{best_marker}"
        )

        # --- Save latest checkpoint ---
        latest_path = os.path.join(checkpoint_dir, "adaptive_scheduler_latest.pt")
        torch.save({
            "epoch": epoch + 1,
            "scheduler_state_dict": scheduler.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "loss_history": loss_history,
            "args": vars(args),
        }, latest_path)

        # --- Save best checkpoint ---
        if is_best:
            best_loss = avg_losses["total_loss"]
            best_path = os.path.join(checkpoint_dir, "adaptive_scheduler_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "scheduler_state_dict": scheduler.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "loss_history": loss_history,
                "args": vars(args),
            }, best_path)

        # --- Plot schedule comparison ---
        plot_schedule_comparison(scheduler, epoch + 1, plots_dir)

    # ------------------------------------------------------------------
    # 8. Post-training
    # ------------------------------------------------------------------
    total_time = time.time() - total_start

    # Final loss curves
    plot_loss_curves(loss_history, plots_dir)

    print(f"-" * 64)
    total_m, total_s = divmod(int(total_time), 60)
    print(f"Done. Best loss: {best_loss:.6f} | Total time: {total_m}m {total_s}s")

    return scheduler


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Adaptive Noise Scheduler for Latent Diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, default="preprocessed_data",
        help="Path to directory containing preprocessed .pt batch files.",
    )
    parser.add_argument(
        "--max_batches", type=int, default=100,
        help="Max number of .pt batch files to load (0 = load all ~3500).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000,
        help="Number of diffusion timesteps T.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to train on ('cuda' or 'cpu'). Auto-detected if not set.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint file to resume training from.",
    )
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


if __name__ == "__main__":
    args = parse_args()
    train_adaptive_scheduler(args)
