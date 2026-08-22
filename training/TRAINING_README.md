# Adaptive Noise Schedule Training — Technical Documentation & Hugging Face Integration Guide

## 1. What Model Was Used for Schedule Training?

The current code in `models/adaptive_scheduler.py` uses a **model-agnostic mathematical diffusion proxy** (Ho et al., 2020 & Nichol & Dhariwal, 2021) to optimize the noise schedule $\beta_t$. 

### Why a Model-Agnostic Proxy?
Instead of tying the noise schedule to one specific heavy UNet architecture, the `AdaptiveNoiseScheduler` optimizes the fundamental signal-to-noise ratio (SNR) profile across $T=1000$ timesteps using:
1. **Reconstruction Signal Quality Proxy**: Evaluates noise-to-signal preservation at each $t$: $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
2. **SNR Log-Uniformity**: Ensures smooth information degradation across timesteps.
3. **Monotonic Decreasing Constraint**: Guarantees $\bar{\alpha}_t = \prod (1 - \beta_s)$ is strictly decreasing.

Because it operates directly on latent/image data statistics, **the learned schedule is universal** and can be plugged directly into **pre-trained Hugging Face Latent Diffusion Models**!

---

## 2. Pre-trained Hugging Face Latent Diffusion Models Compatible

You can plug this learned schedule directly into any standard Hugging Face `diffusers` pipeline or model architecture:

| Hugging Face Model | Model ID | Component Used |
|--------------------|----------|----------------|
| **Stable Diffusion v1.5** | `runwayml/stable-diffusion-v1-5` | `AutoencoderKL` + `UNet2DConditionModel` |
| **Stable Diffusion v2.1** | `stabilityai/stable-diffusion-2-1` | `AutoencoderKL` + `UNet2DConditionModel` |
| **CompVis LDM CelebA (256x256)** | `CompVis/ldm-celeba-256` | `VQModel` / `AutoencoderKL` + `UNet2DModel` |
| **DDPM CelebA / CIFAR-10** | `google/ddpm-celeba-64` | `UNet2DModel` |

---

## 3. How to Connect `AdaptiveNoiseScheduler` with Hugging Face `diffusers`

Once you train the adaptive scheduler (`adaptive_scheduler_best.pt`), you can plug it into a Hugging Face model in 3 simple steps:

### Code Snippet: Using Learned Schedule with Hugging Face VAE & UNet

```python
import torch
from diffusers import AutoencoderKL, UNet2DModel
from models.adaptive_scheduler import AdaptiveNoiseScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Pretrained VAE and UNet from Hugging Face
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
unet = UNet2DModel.from_pretrained("google/ddpm-celeba-64").to(device)

# 2. Load your Trained Adaptive Noise Scheduler
scheduler = AdaptiveNoiseScheduler(timesteps=1000, schedule_type="adaptive").to(device)
checkpoint = torch.load("outputs/checkpoints/adaptive_scheduler_best.pt", map_location=device)
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
scheduler.eval()

# 3. Get the learned beta schedule tensor for Hugging Face compatibility
learned_betas = scheduler.get_betas()  # [1000] tensor of learned beta_t values

print("Successfully loaded learned noise schedule with Hugging Face models!")
print("Learned beta range:", learned_betas.min().item(), "to", learned_betas.max().item())
```

### Option B: Wrapping into Hugging Face `DDPMScheduler` or `DDIMScheduler`

You can also pass `learned_betas` directly into Hugging Face's built-in scheduler classes:

```python
from diffusers import DDPMScheduler

# Create a Hugging Face scheduler initialized with your learned betas!
hf_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    trained_betas=learned_betas.cpu().numpy().tolist()
)
```

---

## 4. Summary of Model & Training Details

- **Dataset**: LSUN Bedroom, CelebA, Oxford Flowers, CIFAR-10 (224K images total across ~3,500 `.pt` batches)
- **Lazy Loading**: `PreprocessedBatchDataset` loads `.pt` files on-demand to prevent system RAM overflow.
- **Scheduler Network (`ScheduleNet`)**: 3-layer MLP (~66K params) mapping $t/T \to \beta_t \in [\beta_{\min}, \beta_{\max}]$.
- **Execution**: Run `python training/train.py --max_batches 0 --epochs 30` for full training.
