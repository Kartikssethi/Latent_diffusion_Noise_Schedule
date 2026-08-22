"""
Adaptive Noise Scheduler for Latent Diffusion Models.

This module implements a trainable noise schedule that learns optimal
beta_t values via a small neural network (MLP), rather than using fixed
linear or cosine schedules. The learned schedule is constrained to produce
valid diffusion parameters:
    - beta_t in (beta_min, beta_max) via sigmoid activation
    - alpha_bar_t is strictly decreasing (cumulative product of 1 - beta_t)

Supports three modes:
    - "linear":   Fixed linear schedule (baseline, not trainable)
    - "cosine":   Fixed cosine schedule (baseline, not trainable)
    - "adaptive": Learnable MLP-parameterized schedule (trainable)

Reference:
    - Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
    - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Baseline Schedule Generators (non-trainable)
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Standard linear beta schedule from DDPM (Ho et al., 2020).
    Returns a 1-D tensor of shape [timesteps].
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine beta schedule from Improved DDPM (Nichol & Dhariwal, 2021).
    Produces a smoother noise ramp that preserves more signal at early steps.
    Returns a 1-D tensor of shape [timesteps].
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]  # normalize so alpha_bar[0] = 1
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, min=1e-5, max=0.999)


# ---------------------------------------------------------------------------
# Trainable Adaptive Schedule Network
# ---------------------------------------------------------------------------

class ScheduleNet(nn.Module):
    """
    Small MLP that maps normalized timestep t/T -> predicted beta_t.

    Architecture:
        Input:  t/T (scalar, normalized to [0, 1])
        Hidden: 2 hidden layers with SiLU activation
        Output: raw logit -> sigmoid -> rescaled to [beta_min, beta_max]

    This ensures beta_t is always in a valid range and the schedule
    can be learned end-to-end via backpropagation.
    """

    def __init__(self, hidden_dim=256, beta_min=1e-4, beta_max=0.02):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize output bias so initial betas ≈ midpoint of range
        # sigmoid(0) = 0.5, so beta ≈ (beta_min + beta_max) / 2
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, t_normalized):
        """
        Args:
            t_normalized: Tensor of shape [N] or [N, 1], values in [0, 1].
        Returns:
            betas: Tensor of shape [N], values in (beta_min, beta_max).
        """
        if t_normalized.dim() == 1:
            t_normalized = t_normalized.unsqueeze(-1)  # [N] -> [N, 1]

        raw = self.net(t_normalized)  # [N, 1]
        # Sigmoid constrains to (0, 1), then rescale to (beta_min, beta_max)
        betas = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(raw)
        return betas.squeeze(-1)  # [N]


# ---------------------------------------------------------------------------
# Main Adaptive Noise Scheduler Module
# ---------------------------------------------------------------------------

class AdaptiveNoiseScheduler(nn.Module):
    """
    Noise scheduler for diffusion models with three modes:
        - "linear":   Fixed linear beta schedule (no trainable params)
        - "cosine":   Fixed cosine beta schedule (no trainable params)
        - "adaptive": Learnable beta schedule via ScheduleNet MLP

    Core functionality:
        - get_betas()       -> full beta_t schedule [T]
        - get_alpha_bar(t)  -> cumulative alpha_bar at timestep t
        - add_noise(x0, t, noise) -> noisy sample x_t
        - compute_schedule_loss(x0) -> training loss for schedule optimization

    The adaptive schedule is trained by optimizing a variational lower bound
    (VLB) proxy: it encourages the learned SNR curve to distribute information
    uniformly across timesteps, matching a target log-SNR profile.
    """

    def __init__(self, timesteps=1000, schedule_type="adaptive",
                 beta_min=1e-4, beta_max=0.02, hidden_dim=256):
        super().__init__()
        self.timesteps = timesteps
        self.schedule_type = schedule_type.lower()
        self.beta_min = beta_min
        self.beta_max = beta_max

        if self.schedule_type == "adaptive":
            # Trainable MLP for predicting beta_t
            self.schedule_net = ScheduleNet(
                hidden_dim=hidden_dim,
                beta_min=beta_min,
                beta_max=beta_max,
            )
        else:
            self.schedule_net = None

        # Pre-compute and register fixed schedules as buffers (non-trainable)
        linear_betas = linear_beta_schedule(timesteps, beta_min, beta_max)
        cosine_betas = cosine_beta_schedule(timesteps)
        self.register_buffer("linear_betas", linear_betas)
        self.register_buffer("cosine_betas", cosine_betas)

        # Pre-compute alpha_bar for fixed schedules
        self.register_buffer("linear_alpha_bar", torch.cumprod(1.0 - linear_betas, dim=0))
        self.register_buffer("cosine_alpha_bar", torch.cumprod(1.0 - cosine_betas, dim=0))

    def get_betas(self):
        """
        Compute the full beta schedule [T].

        For "adaptive", runs the ScheduleNet on all T timesteps.
        For "linear"/"cosine", returns the pre-computed buffer.

        Returns:
            betas: Tensor of shape [T], beta values for each timestep.
        """
        if self.schedule_type == "linear":
            return self.linear_betas

        elif self.schedule_type == "cosine":
            return self.cosine_betas

        elif self.schedule_type == "adaptive":
            # Create normalized timestep inputs: t/T for t in [0, T-1]
            t_normalized = torch.linspace(0, 1, self.timesteps,
                                          device=self.linear_betas.device)
            betas = self.schedule_net(t_normalized)
            return betas

        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

    def get_alpha_bar(self, t=None):
        """
        Compute cumulative product alpha_bar_t = prod_{s=1}^{t} (1 - beta_s).

        Args:
            t: Optional integer tensor of timestep indices [N].
               If None, returns the full alpha_bar curve [T].

        Returns:
            alpha_bar: Tensor of shape [T] (if t is None) or [N] (indexed).
        """
        if self.schedule_type == "linear" and t is not None:
            return self.linear_alpha_bar[t]
        elif self.schedule_type == "cosine" and t is not None:
            return self.cosine_alpha_bar[t]

        # For adaptive (or full curve request), compute from betas
        betas = self.get_betas()
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)

        if t is not None:
            return alpha_bar[t]
        return alpha_bar

    def add_noise(self, x0, t, noise=None):
        """
        Forward diffusion: add noise to clean data x0 at timestep t.

            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x0:    Clean data tensor [B, C, H, W].
            t:     Timestep indices [B], integer values in [0, T-1].
            noise: Optional pre-sampled Gaussian noise [B, C, H, W].
                   If None, sampled from N(0, I).

        Returns:
            x_t:   Noisy data tensor [B, C, H, W].
            noise: The noise that was added [B, C, H, W].
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self.get_alpha_bar(t)  # [B]

        # Reshape for broadcasting: [B] -> [B, 1, 1, 1]
        while alpha_bar_t.dim() < x0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def compute_schedule_loss(self, x0):
        """
        Compute the training loss for optimizing the adaptive noise schedule.

        The loss has three components:
        1. **VLB Reconstruction Loss**: For randomly sampled timesteps, measure
           how well the noisy signal preserves information (proxy via MSE
           between denoised estimate and original).
        2. **SNR Uniformity Loss**: Encourages the log-SNR curve to be
           approximately uniformly spaced (smooth, monotonic decrease).
        3. **Monotonicity Penalty**: Penalizes any violations where alpha_bar
           is not strictly decreasing.

        Args:
            x0: Clean image batch [B, C, H, W], values in [-1, 1].

        Returns:
            total_loss: Scalar loss for backpropagation.
            loss_dict:  Dictionary of individual loss components for logging.
        """
        B = x0.shape[0]
        device = x0.device

        # --- Get the learned schedule ---
        betas = self.get_betas()
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)  # [T]

        # ===================================================================
        # Loss 1: VLB Reconstruction Proxy
        # ===================================================================
        # Sample random timesteps and measure reconstruction quality
        t = torch.randint(0, self.timesteps, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t, _ = self.add_noise(x0, t, noise)

        # Simple denoising estimate: x0_hat = (x_t - sqrt(1-abar)*eps) / sqrt(abar)
        # We use the known noise to compute reconstruction error under the schedule
        abar_t = alpha_bar[t]
        while abar_t.dim() < x0.dim():
            abar_t = abar_t.unsqueeze(-1)

        # Reconstruction: how much signal is preserved at each timestep
        # Higher alpha_bar = more signal preserved = lower reconstruction error
        x0_hat = (x_t - torch.sqrt(1.0 - abar_t) * noise) / (torch.sqrt(abar_t) + 1e-8)
        recon_loss = F.mse_loss(x0_hat, x0)

        # ===================================================================
        # Loss 2: SNR Uniformity (log-SNR should be evenly spaced)
        # ===================================================================
        # log-SNR = log(alpha_bar / (1 - alpha_bar))
        snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
        log_snr = torch.log(snr + 1e-8)  # [T]

        # Ideal: log-SNR decreases uniformly from high to low
        # Compute second derivative (curvature) — should be ~0 for uniform spacing
        log_snr_diff = log_snr[1:] - log_snr[:-1]  # first differences [T-1]
        log_snr_diff2 = log_snr_diff[1:] - log_snr_diff[:-1]  # second differences [T-2]
        uniformity_loss = torch.mean(log_snr_diff2 ** 2)

        # ===================================================================
        # Loss 3: Monotonicity Penalty (alpha_bar must be strictly decreasing)
        # ===================================================================
        # Penalize any increase in alpha_bar (violations of monotonicity)
        alpha_bar_diff = alpha_bar[1:] - alpha_bar[:-1]  # should all be < 0
        monotonicity_loss = torch.mean(F.relu(alpha_bar_diff) ** 2)

        # ===================================================================
        # Loss 4: Boundary Constraints
        # ===================================================================
        # alpha_bar[0] should be close to 1 (almost no noise at t=0)
        # alpha_bar[-1] should be close to 0 (pure noise at t=T)
        boundary_loss = (alpha_bar[0] - 1.0) ** 2 + (alpha_bar[-1] - 0.0) ** 2

        # ===================================================================
        # Combine losses with weights
        # ===================================================================
        total_loss = (
            1.0 * recon_loss
            + 0.1 * uniformity_loss
            + 10.0 * monotonicity_loss
            + 5.0 * boundary_loss
        )

        loss_dict = {
            "recon_loss": recon_loss.item(),
            "uniformity_loss": uniformity_loss.item(),
            "monotonicity_loss": monotonicity_loss.item(),
            "boundary_loss": boundary_loss.item(),
            "total_loss": total_loss.item(),
            "alpha_bar_start": alpha_bar[0].item(),
            "alpha_bar_end": alpha_bar[-1].item(),
            "beta_min_learned": betas.min().item(),
            "beta_max_learned": betas.max().item(),
        }

        return total_loss, loss_dict

    def get_schedule_comparison(self):
        """
        Returns all three schedules (linear, cosine, adaptive) for visualization.

        Returns:
            dict with keys "linear", "cosine", "adaptive", each containing:
                - "betas": [T] tensor
                - "alpha_bar": [T] tensor
        """
        result = {}

        # Linear baseline
        result["linear"] = {
            "betas": self.linear_betas.detach().cpu(),
            "alpha_bar": self.linear_alpha_bar.detach().cpu(),
        }

        # Cosine baseline
        result["cosine"] = {
            "betas": self.cosine_betas.detach().cpu(),
            "alpha_bar": self.cosine_alpha_bar.detach().cpu(),
        }

        # Adaptive (learned)
        if self.schedule_type == "adaptive" and self.schedule_net is not None:
            with torch.no_grad():
                adaptive_betas = self.get_betas().detach().cpu()
                adaptive_alpha_bar = torch.cumprod(1.0 - adaptive_betas, dim=0)
            result["adaptive"] = {
                "betas": adaptive_betas,
                "alpha_bar": adaptive_alpha_bar,
            }

        return result

    def __repr__(self):
        return (
            f"AdaptiveNoiseScheduler(\n"
            f"  schedule_type={self.schedule_type},\n"
            f"  timesteps={self.timesteps},\n"
            f"  beta_range=({self.beta_min}, {self.beta_max}),\n"
            f"  trainable_params={sum(p.numel() for p in self.parameters() if p.requires_grad)}\n"
            f")"
        )
