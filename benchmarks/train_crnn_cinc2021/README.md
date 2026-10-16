# CINC2021: ECG_CRNN benchmark

Benchmark example for the [PhysioNet/Computing in Cardiology Challenge
2021](https://physionetchallenges.org/2021/) (multi-label ECG
classification over the five Challenge lead sets), built directly on
the library: the reader, dataset, trainer machinery, backbones and
challenge metric all come from `torch_ecg`; this folder only contains
what is specific to the challenge.

| file | content |
|---|---|
| `cfg.py` | training config + per-lead-set model configs (systematically generated from the lead count) |
| `model.py` | `ECG_CRNN_CINC2021` — `ECG_CRNN` with the challenge's NSR-fallback inference rule |
| `trainer.py` | `CINC2021Trainer` — `BaseTrainer` wired to `CINC2021Dataset` and the official challenge metric |
| `train.py` | CLI entry |

## Usage

Run from the repository root:

```bash
# 12-lead model on the real database
python -m benchmarks.train_crnn_cinc2021.train --db-dir ~/data/cinc2021 --lead-set twelve_leads

# quick end-to-end check over all five lead sets (2 epochs, debug tranches)
python -m benchmarks.train_crnn_cinc2021.train --db-dir <path> --lead-set all --smoke
```

`train.py` seeds everything through `torch_ecg.utils.seed_everything`
and the trainer reseeds DataLoader workers via `worker_init_fn`, so a
given `--seed` fully determines a run.

The historical contest artifacts (original entry results in `results/`
and the figures in `images/`) are kept for reference.
