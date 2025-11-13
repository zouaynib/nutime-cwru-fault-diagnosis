import functools
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import *
from src.data.transforms import *


# ================================================================
#  Select dataset class and format (.pt for pretrain, .feather for fine-tune)
# ================================================================
def get_dataset_cls(config):
    pt_keywords = ['uea-', 'TFC-']
    data_format = 'feather'
    for kw in pt_keywords:
        if kw in config.dataset or kw in config.dataset_dir:
            data_format = 'pt'
            break
    config.data_format = data_format
    dataset_cls = functools.partial(CustomDataset)
    return dataset_cls, data_format


# ================================================================
#  Data transforms (augmentations / resize)
# ================================================================
def get_transform(config):
    transform_dict = {}
    if config.transform_type == 'rrcrop':
        # augmentation for train
        train_transform = TSCompose([
            TSRandomResizedCrop(size=config.transform_size, scale=config.scale, mode=config.interpolate_mode),
            TSRandomMask(
                scale=config.mask_scale,
                mask_mode=config.mask_mode,
                win_size=config.window_size,
                window_mask_generator=config.window_mask_generator,
            ),
        ])
        # resize for validation and test
        test_transform = TSCompose([
            TSResize(size=config.transform_size, mode=config.interpolate_mode),
        ])
    else:
        # identity transform
        train_transform = TSIdentity()
        test_transform = TSIdentity()
    
    # contrastive two-view transform for SSL
    if config.task == 'ssl':
        train_transform = TSTwoViewsTransform(train_transform)

    transform_dict['train'] = train_transform
    transform_dict['val'] = test_transform
    transform_dict['test'] = test_transform

    return transform_dict


# ================================================================
#  Dataset normalization (global or instance)
# ================================================================
def normalize_dataset(config, dataset_dict):
    eps = 1e-8
    if config.norm == 'global':
        samples = dataset_dict['train'].samples
        mean, std = np.nanmean(samples.numpy(), axis=(0, 2)), np.nanstd(samples.numpy(), axis=(0, 2))
        mean = torch.as_tensor(mean).view(-1, 1)
        std = torch.as_tensor(std).view(-1, 1)
        for dtype in config.dataset_type_list:
            dataset_dict[dtype].samples = (dataset_dict[dtype].samples - mean) / (std + eps)
    elif config.norm == 'instance':
        num_channels = dataset_dict['train'].num_channels
        for dtype in config.dataset_type_list:
            mean = np.nanmean(dataset_dict[dtype].samples.numpy(), axis=(2))
            std = np.nanstd(dataset_dict[dtype].samples.numpy(), axis=(2))
            mean = torch.as_tensor(mean).view(-1, num_channels, 1)
            std = torch.as_tensor(std).view(-1, num_channels, 1)
            dataset_dict[dtype].samples = (dataset_dict[dtype].samples - mean) / (std + eps)


# ================================================================
#  Build dataset dictionary {train, val, test}
# ================================================================
def get_dataset(config):
    transform_dict = get_transform(config)
    dataset_dict = {}
    for dtype in ['train', 'test']:
        dataset_dict[dtype] = CustomDataset(config=config, type=dtype, transform=transform_dict[dtype])

    # Handle missing validation set
    try:
        config.no_validation_set = False
        dataset_dict['val'] = CustomDataset(config=config, type='val', transform=transform_dict['val'])
    except:
        config.no_validation_set = True
        dataset_dict['val'] = CustomDataset(config=config, type='test', transform=transform_dict['test'])

    # Return a validation-like train set for SSL knn evaluation
    if config.task == 'ssl' and config.use_eval:
        dataset_dict['train_knn'] = CustomDataset(config=config, type='train', transform=transform_dict['test'])

    config.dataset_type_list = dataset_dict.keys()
    normalize_dataset(config, dataset_dict)

    return dataset_dict


# ================================================================
#  Weighted sampler (optional)
# ================================================================
def get_weighted_sampler(config, dataset):
    class_counts = torch.bincount(dataset.targets)
    sample_weights = 1 / class_counts[dataset.targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)


# ================================================================
#  Build dataloaders per split
# ================================================================
def build_dataloader(config, dataset_dict):
    dataloader_dict = {}
    for dtype in config.dataset_type_list:
        sampler = None
        if config.use_weighted_sampler and dtype == 'train':
            sampler = get_weighted_sampler(config, dataset_dict[dtype])
        dataloader_dict[dtype] = DataLoader(
            dataset=dataset_dict[dtype],
            shuffle=True if dtype == 'train' and sampler is None else False,
            batch_size=config.batch_size if dtype == 'train' else config.eval_batch_size,
            num_workers=config.num_workers,
            sampler=sampler
        )
    return dataloader_dict


# ================================================================
#  FINAL FIXED FUNCTION: unified class discovery across splits
# ================================================================
def get_dataloader(config):
    # Build datasets and loaders
    dataset_dict = get_dataset(config)
    dataloader_dict = build_dataloader(config, dataset_dict)

    # 🔹 Unify label space across all splits
    all_labels = set()
    for split in ['train', 'val', 'test']:
        if split in dataset_dict and hasattr(dataset_dict[split], 'targets'):
            all_labels.update(dataset_dict[split].targets.cpu().numpy().tolist())
    all_labels = sorted(list(all_labels))

    # Update config
    config.classes = [str(l) for l in all_labels]
    config.num_classes = len(all_labels)
    config.num_channels = dataset_dict['train'].num_channels
    config.ori_series_size = dataset_dict['train'].series_size
    config.model_series_size = config.transform_size
    config.iters_per_epoch = len(dataloader_dict['train'])

    print(f"[DataLoader] Unified total classes: {config.num_classes}")
    print(f"[DataLoader] Channels: {config.num_channels}, Original Series length: {config.ori_series_size}")

    return dataloader_dict
