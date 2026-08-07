from ..dataset_loader import load_dataset

def get_cifar10_dataset(root_dir=None, image_size=256, is_train=True, fraction=0.4):
    return load_dataset("cifar10", root_dir=root_dir, image_size=image_size, is_train=is_train, fraction=fraction)
