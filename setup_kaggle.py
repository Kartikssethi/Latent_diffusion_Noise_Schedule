import sys
import os
import shutil

# 1. Purge cached 'datasets' modules from Python memory
for mod in list(sys.modules.keys()):
    if mod == 'datasets' or mod.startswith('datasets.'):
        del sys.modules[mod]

# 2. Find true repository directory where setup_kaggle.py lives
current_dir = os.path.dirname(os.path.abspath(__file__))

# 3. If nested duplicate folder exists on Kaggle, sync datasets into it to prevent import mismatches
nested_dir = os.path.join(current_dir, "Latent_diffusion_Noise_Schedule")
if os.path.exists(nested_dir):
    src_datasets = os.path.join(current_dir, "datasets")
    dst_datasets = os.path.join(nested_dir, "datasets")
    if os.path.exists(src_datasets) and os.path.exists(dst_datasets):
        try:
            shutil.copytree(src_datasets, dst_datasets, dirs_exist_ok=True)
            print("[setup_kaggle] Synced datasets package into nested directory.")
        except Exception as e:
            print(f"[setup_kaggle] Sync warning: {e}")

# 4. Clean sys.path so current_dir is strictly first
for p in list(sys.path):
    if 'Latent_diffusion' in p or 'Latent_Diffuison' in p:
        sys.path.remove(p)

sys.path.insert(0, current_dir)

print(f"[setup_kaggle] Active repository path set to: '{current_dir}'")
