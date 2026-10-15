# CPSC2019: multi-architecture R-peak detection benchmark

Benchmark example for [CPSC2019](http://2019.icbeb.org/Challenge.html)
(single-lead ECG R-peak detection), with **three architectures** to
compare (hence "multi"): a sequence-labelling CRNN
(`--arch seq_lab_crnn`), a plain UNet (`--arch unet`) and a
subtract-UNet (`--arch subtract_unet`).

Everything generic comes from `torch_ecg` (`CPSC2019Dataset`, the
UNet/seq-lab backbones, and the official R-peak scoring metric); this
folder only contains the CPSC2019-specific inference post-processing
(R-peak extraction from predicted masks) and configs.

| file | content |
|---|---|
| `cfg.py` | training config (from the library's `CPSC2019TrainCfg`) + the three model configs |
| `model.py` | the three CPSC2019 models with R-peak inference |
| `trainer.py` | `CPSC2019Trainer` — `BaseTrainer` wired to `CPSC2019Dataset` and the official scorer |
| `train.py` | CLI entry with `--arch` |

## Usage

Run from the repository root:

```bash
python -m benchmarks.train_multi_cpsc2019.train --db-dir ~/data/cpsc2019 --arch seq_lab_crnn

# quick end-to-end check (2 epochs, debug mode)
python -m benchmarks.train_multi_cpsc2019.train --db-dir <path> --arch seq_lab_crnn --smoke
```

`train.py` seeds everything through `torch_ecg.utils.seed_everything`
and the trainer reseeds DataLoader workers via `worker_init_fn`.
