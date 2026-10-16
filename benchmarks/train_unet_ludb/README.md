# LUDB: ECG_UNET wave-delineation benchmark

Benchmark example for [LUDB](https://physionet.org/content/ludb/1.0.1/)
(ECG wave delineation: P/QRS/T onset & offset detection), built directly
on the library: the reader, dataset (`LUDBDataset`), UNet backbone and
the delineation metrics all come from `torch_ecg`; this folder only
contains what is specific to the task.

Reference: Moskalenko, Zolotykh, Osipov. "Deep Learning for ECG
Segmentation." Neuroinformatics, 2019.

| file | content |
|---|---|
| `cfg.py` | training config (from the library's `LUDBTrainCfg`) + the UNet model config |
| `model.py` | `ECG_UNET_LUDB` — `ECG_UNET` with mask-decoding inference |
| `trainer.py` | `LUDBTrainer` — `BaseTrainer` wired to `LUDBDataset` and the delineation metrics |
| `train.py` | CLI entry |

## Usage

Run from the repository root:

```bash
python -m benchmarks.train_unet_ludb.train --db-dir ~/data/ludb

# quick end-to-end check (2 epochs, debug mode)
python -m benchmarks.train_unet_ludb.train --db-dir <path> --smoke
```

`train.py` seeds everything through `torch_ecg.utils.seed_everything`
and the trainer reseeds DataLoader workers via `worker_init_fn`.
