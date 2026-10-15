"""Training entry for the CinC2023 benchmark example.

The benchmark predicts the neurological recovery (CPC level, or the
binary Good / Poor outcome) of comatose post-cardiac-arrest patients
from bipolar EEG segments (the I-CARE database), using a waveform CRNN.

Examples
--------
# CPC-level (5 classes) prediction on the real database
python -m benchmarks.train_crnn_cinc2023.train --db-dir ~/data/i-care

# binary outcome (Good / Poor) prediction
python -m benchmarks.train_crnn_cinc2023.train --db-dir ~/data/i-care --output-target outcome
"""

import argparse
from copy import deepcopy
from pathlib import Path

from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils import seed_everything

from .cfg import BaseCfg, ModelCfg, TrainCfg
from .model import CRNN_CINC2023
from .trainer import CINC2023Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CRNN on the CinC2023 (I-CARE) database.")
    parser.add_argument("--db-dir", type=str, required=True, help="path to the I-CARE database")
    parser.add_argument(
        "--output-target",
        type=str,
        default=BaseCfg.output_target,
        choices=["cpc", "outcome"],
        help="predict the CPC level (5 classes) or the binary outcome",
    )
    parser.add_argument("--cnn-name", type=str, default=None)
    parser.add_argument("--rnn-name", type=str, default=None, choices=[None, "none", "lstm", "linear"])
    parser.add_argument("--attn-name", type=str, default=None, choices=[None, "none", "se", "gc", "nl"])
    parser.add_argument("--epochs", type=int, default=None, help="overrides TrainCfg.n_epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULTS.SEED)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None, help="where to put logs/checkpoints/models")
    parser.add_argument("--smoke", action="store_true", help="tiny 2-epoch debug run, as in CI")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()

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
    train_config.task = train_config.tasks[0]  # "classification"
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

    task_cfg = train_config.classification
    if args.output_target != task_cfg.output_target:
        task_cfg.output_target = args.output_target
        task_cfg.classes = deepcopy(BaseCfg[args.output_target])
        task_cfg.class_map = deepcopy(BaseCfg[f"{args.output_target}_map"])
    for name in ["cnn_name", "rnn_name", "attn_name"]:
        override = getattr(args, name)
        if override is not None:
            setattr(task_cfg, name, override)
    # fields consumed by the trainer / dataset via the top-level config
    train_config.output_target = task_cfg.output_target
    train_config.classes = deepcopy(task_cfg.classes)
    train_config.class_map = deepcopy(task_cfg.class_map)
    train_config.monitor = task_cfg.monitor
    train_config.final_model_name = task_cfg.final_model_name
    train_config.loss = deepcopy(task_cfg.loss)
    train_config.loss_kw = deepcopy(task_cfg.loss_kw)

    model_config = deepcopy(ModelCfg.classification)
    model_config.output_target = task_cfg.output_target
    model_config.classes = deepcopy(task_cfg.classes)
    model_config.cnn_name = task_cfg.cnn_name
    model_config.rnn_name = task_cfg.rnn_name
    model_config.attn_name = task_cfg.attn_name
    model_config.crnn.cnn.name = task_cfg.cnn_name
    model_config.crnn.rnn.name = task_cfg.rnn_name
    model_config.crnn.attn.name = task_cfg.attn_name

    device = DEFAULTS.device

    model_cls = CRNN_CINC2023
    model_cls.__DEBUG__ = False
    model = model_cls(config=model_config)
    model.to(device=device)

    trainer = CINC2023Trainer(
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
