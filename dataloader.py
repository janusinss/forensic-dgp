from torch.utils.data import DataLoader
from dataset import DegradedFacesDataset
import random
import torch


def split_paths(paths, validation_fraction=0.05, seed=42):
    paths = sorted(set(paths))
    if len(paths) < 2 or not 0 < validation_fraction < 1:
        raise ValueError('Need at least two images and a validation fraction between 0 and 1')
    random.Random(seed).shuffle(paths)
    count = min(len(paths) - 1, max(1, int(len(paths) * validation_fraction)))
    return sorted(paths[count:]), sorted(paths[:count])


def get_training_loaders(root_dir, batch_size=16, num_workers=2, seed=42,
                         validation_fraction=0.05, heavy_blur_probability=0.35, curriculum=True):
    train = DegradedFacesDataset(root_dir, curriculum=curriculum,
                                heavy_blur_probability=heavy_blur_probability, seed=seed)
    val = DegradedFacesDataset(root_dir, curriculum=curriculum,
                              heavy_blur_probability=heavy_blur_probability, seed=seed,
                              extract_landmarks=False)
    train.image_paths, val.image_paths = split_paths(train.image_paths, validation_fraction, seed)
    options = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    # Epoch-specific degradation seeds work with workers recreated each epoch.
    return DataLoader(train, shuffle=True, **options), DataLoader(val, shuffle=False, **options)

def get_dataloader(root_dir, batch_size=8, num_workers=2, shuffle=True, curriculum=True):
    """
    Returns a configured PyTorch DataLoader for the DegradedFacesDataset.
    Optimized for local edge hardware (<8GB VRAM). 
    A batch size of 8-16 is typically recommended for 8GB VRAM when training 
    FPN and GAN models, but can be adjusted based on the specific architecture's memory footprint.
    
    :param root_dir: Directory containing high-res faces.
    :param batch_size: Number of samples per batch.
    :param num_workers: Number of subprocesses for data loading.
    :param shuffle: Whether to shuffle the data at every epoch.
    :param curriculum: Whether to enable multi-scale curriculum downsampling.
    :return: DataLoader instance
    """
    dataset = DegradedFacesDataset(root_dir=root_dir, curriculum=curriculum)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True, # Speeds up transfer to GPU
        drop_last=True
    )
    
    return dataloader
