import sys
import os

# 1. Target repository directory (where setup_kaggle.py is located)
target_repo_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Remove any nested or duplicate repo entries from sys.path
for p in list(sys.path):
    if 'Latent_diffusion' in p or 'Latent_Diffuison' in p:
        sys.path.remove(p)

# 3. Insert target_repo_dir at index 0 of sys.path
sys.path.insert(0, target_repo_dir)

# 4. Purge cached 'datasets' modules from Python memory
for mod in list(sys.modules.keys()):
    if mod == 'datasets' or mod.startswith('datasets.'):
        del sys.modules[mod]

print(f"[setup_kaggle] Active repository path set to: '{target_repo_dir}'")
