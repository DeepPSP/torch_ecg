"""
Test the best-model checkpointing behaviour of `BaseTrainer`.

Regression tests for the bug where the "BestModel" file written by
:meth:`BaseTrainer.train` and its return value contained the weights of
the FINAL epoch instead of the best epoch, because

1. ``state_dict()`` returns references to the parameter tensors, which are
   updated in-place by the optimizer, so the "snapshot" kept drifting, and
2. the BestModel file was serialized at the end of the training, hence
   contained the final weights.

All tests are hermetic (no database downloads).
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.utils_nn import CkptMixin

_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test_trainer_best_model"

# metric schedule whose peak is at epoch 0, so that the best epoch (0)
# differs from the final epoch (2)
_METRIC_SCHEDULE = {0: 0.9, 1: 0.5, 2: 0.3}


class TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return np.random.rand(4).astype(np.float32), np.array([float(index % 2)], dtype=np.float32)


class TinyModel(nn.Module, CkptMixin):
    """Minimal model with a working `save` (provided by `CkptMixin`)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)
        self.config = CFG(hidden=4)

    def forward(self, input):
        return self.lin(input)


class PlainModel(nn.Module):
    """Minimal model WITHOUT `save`, to test the `torch.save` fallback."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, input):
        return self.lin(input)


class TinyTrainer(BaseTrainer):
    def _setup_dataloaders(self, train_dataset=None, val_dataset=None):
        ds = TinyDataset()
        self.train_loader = DataLoader(ds, batch_size=4, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(ds, batch_size=4, collate_fn=self.collate_fn)

    @property
    def batch_dim(self):
        return 0

    @property
    def extra_required_train_config_fields(self):
        return []

    def run_one_step(self, *data):
        input, label = data
        return self.model(input), label

    def evaluate(self, data_loader):
        # keep a frozen copy of the weights at each evaluation for assertions
        if not hasattr(self, "snapshots"):
            self.snapshots = {}
        self.snapshots[self.epoch] = self._snapshot_state_dict()
        return {"metric": _METRIC_SCHEDULE[self.epoch]}

    def _setup_criterion(self):
        self.criterion = nn.MSELoss()

    def train_one_epoch(self, pbar):
        if getattr(self, "crash_at_epoch", None) == self.epoch:
            raise RuntimeError(f"simulated crash at epoch {self.epoch}")
        return super().train_one_epoch(pbar)


def _make_trainer(model=None, **overrides):
    train_config = CFG(
        debug=False,
        monitor="metric",
        n_epochs=3,
        batch_size=4,
        log_step=100,
        optimizer="adam",
        lr_scheduler="none",
        learning_rate=1e-2,
        classes=["x"],
        log_dir=_CWD / "log",
        model_dir=_CWD / "model",
        checkpoints=_CWD / "ckpt",
        keep_checkpoint_max=1,
    )
    train_config.update(overrides)
    return TinyTrainer(
        model=model or TinyModel(),
        dataset_cls=TinyDataset,
        model_config=CFG(hidden=4),
        train_config=train_config,
    )


def _sd_equal(sd_a, sd_b):
    assert set(sd_a.keys()) == set(sd_b.keys()), f"key mismatch: {set(sd_a)} vs {set(sd_b)}"
    return all(torch.equal(sd_a[key], sd_b[key]) for key in sd_a)


@pytest.fixture(autouse=True)
def _prepare_cwd():
    try:
        shutil.rmtree(_CWD)
    except FileNotFoundError:
        pass
    _CWD.mkdir(parents=True, exist_ok=True)
    yield


class TestBestModelCheckpoint:
    def test_best_epoch_weights_are_returned_and_saved(self):
        trainer = _make_trainer()
        returned = trainer.train()

        assert trainer.best_epoch == 0
        assert set(trainer.snapshots.keys()) == {0, 1, 2}
        # the return value must be the best-epoch weights, not the final ones
        assert _sd_equal(returned, trainer.snapshots[0])
        assert not _sd_equal(returned, trainer.snapshots[2])

        # exactly one BestModel artifact must exist, containing the
        # best-epoch weights
        best_files = list((_CWD / "model").glob("BestModel*"))
        assert len(best_files) == 1
        file_sd = load_file(str(best_files[0]))
        assert _sd_equal(file_sd, trainer.snapshots[0])
        assert not _sd_equal(file_sd, trainer.snapshots[2])

    def test_best_model_saved_immediately_on_improvement(self):
        # the training process crashes at epoch 2, AFTER the best epoch (0);
        # the best model must nevertheless already be on disk, since
        # `keep_checkpoint_max` may have deleted the per-epoch checkpoints
        trainer = _make_trainer()
        trainer.crash_at_epoch = 2
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer.train()
        trainer.log_manager.close()  # type: ignore

        best_files = list((_CWD / "model").glob("BestModel*"))
        assert len(best_files) == 1
        file_sd = load_file(str(best_files[0]))
        assert _sd_equal(file_sd, trainer.snapshots[0])

    def test_monitor_none_returns_final_weights(self):
        trainer = _make_trainer(monitor=None)
        returned = trainer.train()
        assert _sd_equal(returned, trainer.snapshots[2])

    def test_model_without_save_falls_back_to_torch_save(self):
        trainer = _make_trainer(model=PlainModel())
        returned = trainer.train()

        assert trainer.best_epoch == 0
        assert _sd_equal(returned, trainer.snapshots[0])

        best_files = list((_CWD / "model").glob("BestModel*"))
        assert len(best_files) == 1
        ckpt = torch.load(best_files[0], map_location="cpu")
        assert _sd_equal(ckpt["model_state_dict"], trainer.snapshots[0])
