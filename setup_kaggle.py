import sys
import os

# Unload any pre-cached Hugging Face 'datasets' package from sys.modules
sys.modules.pop('datasets', None)

# Add repository root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)
sys.path.insert(0, current_dir)

print(f"[setup_kaggle] Added repository root '{current_dir}' to sys.path & resolved module collisions.")
