"""Training entry for the CPSC2019 R-peak detection benchmark example.

Three architectures are available for comparison (hence "multi"):
a sequence-labelling CRNN, a plain UNet, and a subtract-UNet.

Run from the repository root, e.g.:

python -m benchmarks.train_multi_cpsc2019.train --db-dir ~/data/cpsc2019 --arch seq_lab_crnn
"""

import argparse
from copy import deepcopy
from pathlib import Path

from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils import seed_everything

from .cfg import ModelCfg, TrainCfg
from .model import ECG_SEQ_LAB_NET_CPSC2019, ECG_SUBTRACT_UNET_CPSC2019, ECG_UNET_CPSC2019
from .trainer import CPSC2019Trainer

_ARCH_TO_MODEL = {
    "seq_lab_crnn": ECG_SEQ_LAB_NET_CPSC2019,
    "unet": ECG_UNET_CPSC2019,
    "subtract_unet": ECG_SUBTRACT_UNET_CPSC2019,
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an R-peak detector on CPSC2019.")
    parser.add_argument("--db-dir", type=str, required=True, help="path to the CPSC2019 database")
    parser.add_argument(
        "--arch",
        type=str,
        default="seq_lab_crnn",
        choices=list(_ARCH_TO_MODEL),
        help="model architecture to train",
    )
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

    model_config = deepcopy(ModelCfg[args.arch])

    model = _ARCH_TO_MODEL[args.arch](
        n_leads=train_config.n_leads,
        config=model_config,
    )
    model.to(device=device)

    trainer = CPSC2019Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        lazy=False,
    )
    best_model_dict = trainer.train()

    return {args.arch: best_model_dict}


if __name__ == "__main__":
    main()
