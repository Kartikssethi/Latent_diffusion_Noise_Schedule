import sys
import os

# Unload any pre-cached 'datasets' package or submodules from sys.modules
for mod in list(sys.modules.keys()):
    if mod == 'datasets' or mod.startswith('datasets.'):
        del sys.modules[mod]

# Add repository root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)
sys.path.insert(0, current_dir)

print(f"[setup_kaggle] Cleared cached 'datasets' modules & set sys.path to '{current_dir}'.")
