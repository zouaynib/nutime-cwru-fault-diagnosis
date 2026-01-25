# Simple helper to prepare CWRU dataset for NuTime
# Saves train.pt, val.pt, test.pt under out_dir.
#
# Usage:
#   python tools/prepare_cwru.py --raw_dir /path/to/raw_cwru --out_dir ./datasets/CWRU --length 2048 --val_ratio 0.1 --test_ratio 0.1
#
# The script expects raw_dir to have subfolders per class, with files inside that are .npy, .mat, or .txt.
# Each sample will be padded/truncated to `length` and stored as shape (1, length) for single-channel signals.

import os
import argparse
from glob import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import scipy.io

def load_signal(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
    elif ext == '.mat':
        mat = scipy.io.loadmat(path)
        # Heuristic: pick the first array variable that is numeric and 1D/2D
        arr = None
        for k, v in mat.items():
            if k.startswith('__'): continue
            if isinstance(v, np.ndarray) and v.size > 0:
                arr = np.asarray(v).squeeze()
                break
        if arr is None:
            raise ValueError(f"Cannot find signal array in {path}")
    elif ext in ('.txt', '.csv'):
        arr = np.loadtxt(path, delimiter=',' if ext=='.csv' else None)
    else:
        raise ValueError(f"Unsupported extension {ext} for {path}")
    # ensure 1D
    arr = np.asarray(arr).astype(np.float32).squeeze()
    if arr.ndim > 1:
        # If multi-channel, try to collapse to 1D by taking first row/col if sensible
        arr = arr.reshape(-1)
    return arr

def pad_or_truncate(x, length):
    if len(x) > length:
        return x[:length]
    elif len(x) < length:
        pad = np.zeros(length - len(x), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)
    else:
        return x

def collect_samples(raw_dir, length):
    classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    samples = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(raw_dir, cls)
        # find supported files
        files = []
        for ext in ('*.npy', '*.mat', '*.txt', '*.csv'):
            files.extend(glob(os.path.join(cls_dir, ext)))
        if len(files) == 0:
            print(f"[WARN] No files found for class {cls} in {cls_dir}")
            continue
        for f in files:
            sig = load_signal(f)
            sig = pad_or_truncate(sig, length)
            # convert to shape (1, length) single-channel
            samples.append(sig.reshape(1, -1))
            labels.append(idx)
    samples = np.stack(samples, axis=0)  # (N, C, L)
    labels = np.array(labels, dtype=np.int64)
    return samples, labels, classes

def save_splits(samples, labels, classes, out_dir, val_ratio, test_ratio, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    # First split off test
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=test_ratio,
        random_state=seed,
        stratify=labels
    )
    # Now split train into train+val
    rel_val = val_ratio / (1.0 - test_ratio) if (1.0 - test_ratio) > 0 else 0.0
    train_idx2, val_idx = train_test_split(
        train_idx,
        test_size=rel_val,
        random_state=seed,
        stratify=labels[train_idx] if len(train_idx)>0 else None
    )
    # helper to save
    def save_subset(idxs, filename):
        obj = {
            'samples': torch.tensor(samples[idxs], dtype=torch.float32),
            'labels': torch.tensor(labels[idxs], dtype=torch.long)
        }
        torch.save(obj, os.path.join(out_dir, filename))
    save_subset(train_idx2, 'train.pt')
    save_subset(val_idx, 'val.pt')
    save_subset(test_idx, 'test.pt')
    # Save class mapping
    with open(os.path.join(out_dir, 'classes.txt'), 'w') as f:
        for i, c in enumerate(classes):
            f.write(f"{i}\t{c}\n")
    print(f"Saved train/val/test .pt to {out_dir}. Classes saved to classes.txt")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir', required=True, help='Raw CWRU root dir (subfolders per class)')
    p.add_argument('--out_dir', required=True, help='Output dataset dir (e.g., ./datasets/CWRU)')
    p.add_argument('--length', type=int, default=2048, help='Target length per sample (pad/truncate)')
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--test_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    samples, labels, classes = collect_samples(args.raw_dir, args.length)
    print(f"Collected {len(labels)} samples across {len(classes)} classes, sample shape {samples.shape}")
    save_splits(samples, labels, classes, args.out_dir, args.val_ratio, args.test_ratio, args.seed)