import sys
import os
import shutil

# 1. Purge ALL cached 'datasets' modules from Python memory
for mod in list(sys.modules.keys()):
    if mod == 'datasets' or mod.startswith('datasets.'):
        del sys.modules[mod]

# 2. Find the real repo root: the directory that contains 'datasets/preprocessing.py'
def find_repo_root():
    candidates = [
        '/kaggle/working/Latent_diffusion_Noise_Schedule',
        '/kaggle/working/Latent_diffusion_Noise_Schedule/Latent_diffusion_Noise_Schedule',
        '/kaggle/working',
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd(),
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, 'datasets', 'preprocessing.py')):
            return path
    return None

repo_root = find_repo_root()

if repo_root is None:
    raise RuntimeError("[setup_kaggle] Could not find repo root. Check your Kaggle directory structure.")

# 3. If there is a nested duplicate, copy the latest datasets package into it
nested = os.path.join(repo_root, 'Latent_diffusion_Noise_Schedule')
if os.path.isdir(nested):
    src = os.path.join(repo_root, 'datasets')
    dst = os.path.join(nested, 'datasets')
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"[setup_kaggle] Synced latest datasets/ into nested dir: {nested}")

# 4. Remove any stale repo paths from sys.path and insert the correct root first
for p in list(sys.path):
    if 'Latent_diffusion' in p or 'Latent_Diffuison' in p:
        sys.path.remove(p)
sys.path.insert(0, repo_root)

print(f"[setup_kaggle] Repository root: '{repo_root}'")
print(f"[setup_kaggle] sys.path[0] = '{sys.path[0]}'")
