# NuTime — Numerically Multi-Scaled Embedding for Time-Series Pretraining x CWRU Dataset

[![arXiv](https://img.shields.io/badge/arXiv-2310.07402-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2310.07402)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

This fork/adaptation packages NuTime and includes demo configurations for fault diagnosis (CWRU) and other time-series tasks (e.g., Epilepsy). NuTime provides a transformer-based WinT backbone and the WindowNormEncoder to create multi-scaled numeric embeddings for efficient and effective time-series pretraining and finetuning.


NuTime: Numerically Multi-Scaled Embedding for Large-Scale Time-Series Pretraining — Chenguo Lin et al. (TMLR 2024)  
Paper: https://arxiv.org/abs/2310.07402



CWRU (Case Western Reserve University) demo — dataset preparation 
-----------------------------------------------------------------

NuTime's data loader (see `src/data/dataset.py` and `src/data/dataloader.py`) expects each dataset to be available under:
`<dataset_root>/<dataset_name>/`

Within the dataset folder NuTime will look for either:
- PyTorch `.pt` files: `train.pt`, `val.pt`, `test.pt` — OR —
- Feather `.feather` files: `train.feather`, `val.feather`, `test.feather`

Which format is easiest depends on your raw CWRU data. The recommended (and simplest) format for CWRU is the `.pt` format described below.

.pt format expected by NuTime
- Each file is a dict saved with `torch.save(dict, path)` where:
  - `dict['samples']` is a torch.Tensor of shape (N, C, L)
    - N = number of samples
    - C = number of channels (1 for typical CWRU single-channel vibration signals)
    - L = length (number of time samples per instance)
  - `dict['labels']` is a torch.LongTensor of shape (N,)
    - class indices (0, 1, 2, ...) for each sample
- When `CustomDataset` loads a `.pt` file it calls `data['labels'].long()` and `data['samples'].float()`; NuTime will automatically set `config.classes` from unique labels in the `.pt` file.
- Save files as:
  - `datasets/CWRU/train.pt`
  - `datasets/CWRU/val.pt`
  - `datasets/CWRU/test.pt`

.feather format (alternative)
- A `.feather` file should have the first column as the class label (strings are fine) and subsequent columns the time-series values flattened into columns. The repository already contains helpers for creating `.feather` in `src/data/preprocess.py`.
- Save files as:
  - `datasets/CWRU/train.feather`
  - `datasets/CWRU/val.feather`
  - `datasets/CWRU/test.feather`

Recommended workflow to prepare CWRU for NuTime
1. Collect/organize raw CWRU signals:
   - Option A (recommended): Organize raw samples as per-class directories where each sample is a .npy file (1D array) or a .mat file that can be loaded into a numpy array.
     Example:
       raw_cwru/
         ├─ class_0/
         │    ├─ sample_000.npy
         │    ├─ sample_001.npy
         │    └─ ...
         ├─ class_1/
         │    └─ ...
         └─ class_2/
   - Option B: If you already have large monolithic recordings, segment them into fixed-length windows and label them accordingly.

2. Use the helper script to:
   - load samples,
   - optionally resample/trim/pad to a fixed length L,
   - assign integer labels,
   - split into train / val / test (stratified), and
   - save `train.pt`, `val.pt`, and `test.pt` in `datasets/CWRU/`.

3. Update the demo config:
   - `configs/demo_ft_cwru.json` includes `"dataset": "CWRU"` and `"dataset_dir": "./datasets"`. Ensure your files are under `./datasets/CWRU/` (relative to `NuTime` root).
   - Check `transform_size`/`window_size` in config — these control how transforms are applied. If your samples have length L, ensure `config.transform_size` and `config.window_size` are compatible (or set `transform_size_mode: "auto"`).

Example helper script (converts raw class folders into train/val/test .pt files)
- A convenience script is included in `tools/prepare_cwru.py` (example below). It:
  - walks a `raw_dir`, assumes subfolders are class names,
  - loads `.npy` or `.mat` files (or `.txt` with numeric arrays),
  - pads/truncates each sample to a target length `L`,
  - produces stratified train/val/test splits,
  - saves `train.pt`, `val.pt`, and `test.pt` under `datasets/CWRU/`.

Running the script (example)
1. Place your raw files under `raw_cwru/` (subfolders per class).
2. From `NuTime` root:
   python tools/prepare_cwru.py --raw_dir ../raw_cwru --out_dir ./datasets/CWRU --length 2048 --val_ratio 0.1 --test_ratio 0.1

3. Confirm files exist:
   ls datasets/CWRU
   # should show train.pt val.pt test.pt

Run the demo finetune on CWRU
- After preparing the files:
  python3 src/pipeline.py --config_file configs/demo_ft_cwru.json

Implementation notes / why this works
- `src/data/dataset.py`'s `CustomDataset` first tries `<dataset_dir>/<dataset>/train.pt` and falls back to `.feather`.
- `src/data/dataloader.py` builds `train`, `val`, and `test` loaders and unifies label space across splits, then updates `config.classes`, `config.num_classes`, and `config.num_channels` automatically.
- If your dataset is multi-channel (e.g., C=3), make sure the samples have shape (N, C, L). For single-channel signals, shape is (N, 1, L).

