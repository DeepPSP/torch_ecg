"""Training entry for the LUDB wave-delineation benchmark example.

Run from the repository root, e.g.:

python -m benchmarks.train_unet_ludb.train --db-dir ~/data/ludb --smoke
"""

import argparse
from copy import deepcopy
from pathlib import Path

from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils import seed_everything

from .cfg import ModelCfg, TrainCfg
from .model import ECG_UNET_LUDB
from .trainer import LUDBTrainer


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ECG_UNET on LUDB (wave delineation).")
    parser.add_argument("--db-dir", type=str, required=True, help="path to the LUDB database")
    parser.add_argument("--epochs", type=int, default=None, help="overrides TrainCfg.n_epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULTS.SEED)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None, help="where to put logs/checkpoints/models")
    parser.add_argument("--smoke", action="store_true", help="tiny 2-epoch debug run")
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

    device = DEFAULTS.device

    model_config = deepcopy(ModelCfg)

    model = ECG_UNET_LUDB(
        n_leads=train_config.n_leads,
        config=model_config,
    )
    model.to(device=device)

    trainer = LUDBTrainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        lazy=False,
    )
    best_model_dict = trainer.train()

    return {"unet": best_model_dict}


if __name__ == "__main__":
    main()
