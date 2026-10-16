# CINC2020: ECG_CRNN benchmark

Benchmark example for the [PhysioNet/Computing in Cardiology Challenge
2020](https://physionetchallenges.org/2020/) (12-lead multi-label ECG
classification).

CINC2020's data is a **strict subset of CINC2021's** and the challenge
is 12-lead only, so this benchmark is a thin variant of the
[CINC2021 benchmark](../train_crnn_cinc2021): the model and the trainer
machinery are imported from there; only the dataset (`CINC2020Dataset`)
and the official metric (the 2020 challenge metric, without the 2021
load-balancing multiplier) differ.

## Usage

Run from the repository root:

```bash
python -m benchmarks.train_crnn_cinc2020.train --db-dir ~/data/cinc2020

# quick end-to-end check (2 epochs, debug tranches)
python -m benchmarks.train_crnn_cinc2020.train --db-dir <path> --smoke
```
