"""Training entry for the CinC2022 benchmark example.

The benchmark detects heart murmurs from phonocardiograms (PCG) of the
Circulatory Sounds database, with three training tasks:

1. ``classification`` -- murmur classification (with an optional
   clinical outcome head), via CRNN or wav2vec2 backbones;
2. ``segmentation`` -- heart-state sequence segmentation (S1, systole,
   S2, diastole), via sequence labelling or U-Net;
3. ``multi_task`` -- the CRNN with murmur classification, outcome and
   segmentation heads trained jointly.

Examples
--------
# murmur classification task, on the real database
python -m benchmarks.train_mtl_cinc2022.train --db-dir ~/data/circulatory-sounds --task classification

# heart-state segmentation task, with the U-Net model
python -m benchmarks.train_mtl_cinc2022.train --db-dir ~/data/circulatory-sounds --task segmentation --model-name unet

# the multi-task model (all heads)
python -m benchmarks.train_mtl_cinc2022.train --db-dir ~/data/circulatory-sounds --task multi_task
"""

import argparse
from copy import deepcopy
from pathlib import Path

from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils import seed_everything

from .cfg import ModelCfg, TrainCfg
from .models import CRNN_CINC2022, SEQ_LAB_NET_CINC2022, UNET_CINC2022, HFWav2Vec2_CINC2022, Wav2Vec2_CINC2022
from .trainer import CINC2022Trainer

_MODEL_MAP = {
    "wav2vec2": Wav2Vec2_CINC2022,
    "wav2vec2_hf": HFWav2Vec2_CINC2022,
    "crnn": CRNN_CINC2022,
    "seq_lab": SEQ_LAB_NET_CINC2022,
    "unet": UNET_CINC2022,
}

_TASK_MODEL_NAMES = {
    "classification": ["crnn", "wav2vec2", "wav2vec2_hf"],
    "segmentation": ["seq_lab", "unet"],
    "multi_task": ["crnn"],
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the models on the CinC2022 (Circulatory Sounds) database.")
    parser.add_argument("--db-dir", type=str, required=True, help="path to the Circulatory Sounds database")
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=list(TrainCfg.tasks) + ["all"],
        help="which task to train (or all of them in sequence)",
    )
    parser.add_argument("--model-name", type=str, default=None, help="override the default model of the (single) task")
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
    return parser.parse_args(argv)


def _set_task(task: str, config: dict) -> None:
    """Set the fields of `config` that vary per task (classes, monitor, loss, etc.)."""
    assert task in config.tasks
    config.task = task
    for item in [
        "classes",
        "monitor",
        "final_model_name",
        "loss",
    ]:
        config[item] = config[task][item]


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

    if args.model_name is not None and args.task == "all":
        raise ValueError("--model-name is only valid together with a single --task")

    tasks = list(train_config.tasks) if args.task == "all" else [args.task]
    device = DEFAULTS.device

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
            if override is not None:
                setattr(task_cfg, name, override)

        _set_task(task, train_config)

        model_config = deepcopy(ModelCfg[task])
        model_config.model_name = task_cfg.model_name
        model_config.cnn_name = task_cfg.cnn_name
        model_config.rnn_name = task_cfg.rnn_name
        model_config.attn_name = task_cfg.attn_name
        for arch_name in ["crnn", "seq_lab"]:
            if arch_name in model_config:
                model_config[arch_name].cnn.name = task_cfg.cnn_name
                model_config[arch_name].rnn.name = task_cfg.rnn_name
                model_config[arch_name].attn.name = task_cfg.attn_name

        model_cls = _MODEL_MAP[model_config.model_name]
        model_cls.__DEBUG__ = False
        model = model_cls(config=model_config)
        model.to(device=device)

        trainer = CINC2022Trainer(
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
