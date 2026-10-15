"""Training entry for the CPSC2021 benchmark example.

The benchmark is a 3-task pipeline over long-term 2-lead ECGs:

1. ``qrs_detection`` -- ECG_SEQ_LAB_NET (or ECG_UNET) R-peak detection,
   scored with the CPSC2019 challenge metric;
2. ``rr_lstm`` -- RR_LSTM (optionally with a CRF layer) classifying AF
   on RR interval sequences;
3. ``main`` -- ECG_SEQ_LAB_NET (or ECG_UNET) segmenting AF episodes
   directly from the ECG.

Examples
--------
# train the full pipeline (all three tasks in sequence)
python -m benchmarks.train_hybrid_cpsc2021.train --db-dir ~/data/cpsc2021

# train a single task, with a non-default architecture
python -m benchmarks.train_hybrid_cpsc2021.train --db-dir ~/data/cpsc2021 \
    --task main --model-name unet

# quick smoke run (2 epochs, debug logging) as used in CI
python -m benchmarks.train_hybrid_cpsc2021.train --db-dir <path> --task qrs_detection --smoke
"""

import argparse
from copy import deepcopy
from pathlib import Path

from torch_ecg.cfg import DEFAULTS
from torch_ecg.databases.datasets.cpsc2021 import CPSC2021Dataset
from torch_ecg.utils import seed_everything

from .cfg import ModelCfg, TrainCfg
from .trainer import _MODEL_MAP, CPSC2021Trainer, _set_task

_TASK_MODEL_NAMES = {
    "qrs_detection": ["seq_lab", "unet"],
    "rr_lstm": ["lstm", "lstm_crf"],
    "main": ["seq_lab", "unet"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid models on CPSC2021.")
    parser.add_argument("--db-dir", type=str, required=True, help="path to the CPSC2021 database")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=list(TrainCfg.tasks) + ["all"],
        help="which task to train (or all of them in sequence)",
    )
    parser.add_argument("--model-name", type=str, default=None, help="override the default model of the (single) task")
    parser.add_argument("--cnn-name", type=str, default=None, help="cnn of the seq_lab model (qrs_detection / main)")
    parser.add_argument("--rnn-name", type=str, default=None, choices=[None, "none", "lstm"])
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

    if args.model_name is not None and args.task == "all":
        raise ValueError("--model-name is only valid together with a single --task")

    tasks = list(train_config.tasks) if args.task == "all" else [args.task]
    device = DEFAULTS.device

    # offline preparation: bandpass the records, slice them into segments
    # and generate RR sequences (idempotent, skipped if already done);
    # the same sliced data is also distributed as
    # https://www.kaggle.com/wenh06/cpsc2021-sliced
    ds = CPSC2021Dataset(config=train_config, task=tasks[0], training=True, lazy=True)
    ds.persistence(verbose=1)
    del ds

    best_model_dicts = {}
    for task in tasks:
        task_cfg = train_config[task]
        if args.model_name is not None:
            assert args.model_name in _TASK_MODEL_NAMES[task], (
                f"{args.model_name!r} is not a valid model for task {task!r}, " f"valid ones: {_TASK_MODEL_NAMES[task]}"
            )
            task_cfg.model_name = args.model_name
        for name in ["cnn_name", "rnn_name", "attn_name"]:
            override = getattr(args, name)
            if override is not None and getattr(task_cfg, name, None) is not None:
                setattr(task_cfg, name, override)

        model_config = deepcopy(ModelCfg[task])
        model_config.model_name = task_cfg.model_name
        if model_config.model_name == "seq_lab":
            model_config.cnn_name = task_cfg.cnn_name
            model_config.rnn_name = task_cfg.rnn_name
            model_config.attn_name = task_cfg.attn_name
            model_config.seq_lab.cnn.name = task_cfg.cnn_name
            model_config.seq_lab.rnn.name = task_cfg.rnn_name
            model_config.seq_lab.attn.name = task_cfg.attn_name

        _set_task(task, train_config)

        model_cls = _MODEL_MAP[model_config.model_name]
        model_cls.__DEBUG__ = False
        model = model_cls(config=model_config)
        model.to(device=device)

        trainer = CPSC2021Trainer(
            model=model,
            model_config=model_config,
            train_config=train_config,
            device=device,
            lazy=False,
        )
        best_model_dicts[task] = trainer.train()

        del trainer, model

    return best_model_dicts


if __name__ == "__main__":
    main()
