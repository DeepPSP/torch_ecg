"""Training entry for the CPSC2020 benchmark example.

The benchmark detects premature beats (SPB / PVC) from long-term
single-lead ECGs, with two model families:

- ``crnn`` -- segment-level classification (N / S / V), evaluated by
  the official AUROC / AUPRC / accuracy / F-measure / F-beta / G-beta;
- ``seq_lab`` -- beat-level sequence labelling (S / V localization),
  evaluated by the official challenge score from the library.

Examples
--------
# segment-level classification model on the real database
python -m benchmarks.train_hybrid_cpsc2020.train --db-dir ~/data/cpsc2020

# beat-level sequence labelling model
python -m benchmarks.train_hybrid_cpsc2020.train --db-dir ~/data/cpsc2020 --model-name seq_lab

# quick smoke run (2 epochs, debug logging) as used in CI
python -m benchmarks.train_hybrid_cpsc2020.train --db-dir <path> --smoke
"""

import argparse
from copy import deepcopy
from pathlib import Path

from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils import seed_everything

from .cfg import ModelCfg, TrainCfg
from .dataset import CPSC2020
from .model import ECG_CRNN_CPSC2020, ECG_SEQ_LAB_NET_CPSC2020
from .trainer import CPSC2020Trainer

_MODEL_MAP = {
    "crnn": ECG_CRNN_CPSC2020,
    "seq_lab": ECG_SEQ_LAB_NET_CPSC2020,
}

_MODEL_MONITOR = {
    "crnn": "f_beta_measure",
    "seq_lab": "challenge_score",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the models on CPSC2020.")
    parser.add_argument("--db-dir", type=str, required=True, help="path to the CPSC2020 database")
    parser.add_argument("--model-name", type=str, default="crnn", choices=list(_MODEL_MAP))
    parser.add_argument("--cnn-name", type=str, default="multi_scopic")
    parser.add_argument("--rnn-name", type=str, default="linear")
    parser.add_argument("--epochs", type=int, default=None, help="overrides TrainCfg.n_epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULTS.SEED)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None, help="where to put logs/checkpoints/models")
    parser.add_argument("--smoke", action="store_true", help="tiny 2-epoch debug run, as in CI")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)

    # ONE seeding point for everything (python/numpy/torch/DEFAULTS.RNG);
    # DataLoader workers are reseeded via `worker_init_fn` in the trainer
    seed_everything(args.seed)

    train_config = deepcopy(TrainCfg)
    train_config.db_dir = Path(args.db_dir).expanduser().resolve()
    if args.outdir is not None:
        outdir = Path(args.outdir).expanduser().resolve()
        train_config.log_dir = outdir / "logs"
        train_config.model_dir = outdir / "saved_models"
        train_config.checkpoints = outdir / "checkpoints"
    train_config.debug = args.smoke
    if args.epochs is not None:
        train_config.n_epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.lr is not None:
        train_config.learning_rate = args.lr
    if args.num_workers is not None:
        train_config.num_workers = args.num_workers
    if args.smoke:
        train_config.n_epochs = 2
        train_config.keep_checkpoint_max = 0

    model_name = args.model_name.lower()
    train_config.model_name = model_name
    train_config.classes = deepcopy(ModelCfg[model_name].classes)
    train_config.class_map = deepcopy(ModelCfg[model_name].class_map)
    train_config.monitor = _MODEL_MONITOR[model_name]

    model_config = deepcopy(ModelCfg[model_name])
    model_config.cnn_name = args.cnn_name
    model_config.cnn.name = args.cnn_name
    if model_name == "crnn":
        model_config.rnn_name = args.rnn_name
        model_config.rnn.name = args.rnn_name

    device = DEFAULTS.device

    # offline preparation: slice the records into 10 s segments with
    # overlap around premature beats (idempotent, skipped if already done)
    ds = CPSC2020(config=train_config, training=True, prepare=True)
    ds.persistence(verbose=1)
    del ds

    model_cls = _MODEL_MAP[model_name]
    model_cls.__DEBUG__ = False
    model = model_cls(
        classes=train_config.classes,
        n_leads=train_config.n_leads,
        config=model_config,
    )
    model.to(device=device)

    trainer = CPSC2020Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        lazy=False,
    )
    best_model_state_dict = trainer.train()

    return best_model_state_dict


if __name__ == "__main__":
    main()
