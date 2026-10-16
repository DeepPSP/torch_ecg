"""Training entry for the CINC2021 benchmark example.

Examples
--------
# full 12-lead training on the real database
python train.py --db-dir ~/data/cinc2021 --lead-set twelve_leads

# quick smoke run (2 epochs, tiny tranches) as used in CI
python train.py --db-dir <path> --lead-set all --smoke
"""

import argparse
from copy import deepcopy
from pathlib import Path

from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils import seed_everything

from .cfg import LEAD_SETS, ModelCfg, TrainCfg
from .model import ECG_CRNN_CINC2021
from .trainer import CINC2021Trainer


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ECG_CRNN on CINC2021.")
    parser.add_argument("--db-dir", type=str, required=True, help="path to the CINC2021 database")
    parser.add_argument(
        "--lead-set",
        type=str,
        default="twelve_leads",
        choices=list(LEAD_SETS) + ["all"],
        help="which Challenge lead set to train (or all of them in sequence)",
    )
    parser.add_argument("--cnn-name", type=str, default="resnet_nature_comm_bottle_neck_se")
    parser.add_argument("--rnn-name", type=str, default="none", choices=["none", "lstm", "linear"])
    parser.add_argument("--attn-name", type=str, default="none", choices=["none", "se", "gc", "nl"])
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
    train_config.cnn_name = args.cnn_name
    train_config.rnn_name = args.rnn_name
    train_config.attn_name = args.attn_name
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

    tranches = train_config.tranches_for_training
    train_classes = train_config.tranche_classes[tranches] if tranches else train_config.classes

    device = DEFAULTS.device

    lead_set_names = list(LEAD_SETS) if args.lead_set == "all" else [args.lead_set]

    best_model_dicts = {}
    for lead_set_name in lead_set_names:
        train_config.leads = LEAD_SETS[lead_set_name]
        train_config.n_leads = len(train_config.leads)
        model_config = deepcopy(ModelCfg[lead_set_name])
        model_config.cnn.name = train_config.cnn_name
        model_config.rnn.name = train_config.rnn_name
        model_config.attn.name = train_config.attn_name

        model = ECG_CRNN_CINC2021(
            classes=train_classes,
            n_leads=train_config.n_leads,
            config=model_config,
        )
        model.to(device=device)

        trainer = CINC2021Trainer(
            model=model,
            model_config=model_config,
            train_config=train_config,
            device=device,
            lazy=False,
        )
        best_model_dicts[lead_set_name] = trainer.train()

        del trainer, model

    return best_model_dicts


if __name__ == "__main__":
    main()
