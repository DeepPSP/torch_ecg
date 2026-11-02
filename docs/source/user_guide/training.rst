.. _ug_training:

Training
========

Overview
--------

:class:`~torch_ecg.components.trainer.BaseTrainer` implements the
training loop (epochs, device handling, criterion setup, logging,
early stopping, checkpoint retention, resume); the subclass defines
what is trained: the datasets, one optimization step, and the
evaluation.  The
complete reference implementation is
`benchmarks/train_crnn_cinc2021/trainer.py
<https://github.com/DeepPSP/torch_ecg/blob/master/benchmarks/train_crnn_cinc2021/trainer.py>`__,
from which this page is condensed.

Anatomy of a trainer subclass
-----------------------------

.. code-block:: python

    from torch_ecg.components.trainer import BaseTrainer

    class CINC2021Trainer(BaseTrainer):
        __name__ = "CINC2021Trainer"
        _dataset_cls = CINC2021Dataset      # used by _setup_dataloaders

        def _setup_dataloaders(self, train_dataset=None, val_dataset=None):
            if train_dataset is None:
                train_dataset = self.dataset_cls(config=self.train_config, training=True, lazy=False)
            if val_dataset is None:
                val_dataset = self.dataset_cls(config=self.train_config, training=False, lazy=False)
            ...                               # build DataLoaders with worker_init_fn

        def run_one_step(self, *data):
            signals, labels = data
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(signals)
            return preds, labels             # the loop applies self.criterion(preds, labels)

        @torch.no_grad()
        def evaluate(self, data_loader):
            ...                              # the inference loop (self._model.inference(...))
            return {"weighted_f1": ...}      # keys must contain the monitored metric

The train config
----------------

A train config is a plain :class:`~torch_ecg.cfg.CFG`; the required
fields are ``classes``, ``n_epochs``, ``batch_size``, ``log_step``,
``optimizer``, ``lr_scheduler``, ``learning_rate`` (plus any
``extra_required_train_config_fields`` defined by the task, e.g.
``db_dir``).
Start from a library ``TrainCfg``, adapt, inherit the rest:

.. code-block:: python

    from copy import deepcopy
    from torch_ecg.databases.datasets.cinc2021 import CINC2021TrainCfg

    TrainCfg = deepcopy(CINC2021TrainCfg)
    TrainCfg.db_dir = "~/data/cinc2021"
    TrainCfg.n_epochs = 30
    TrainCfg.monitor = "weighted_f1"     # early stopping + checkpoint selection
    TrainCfg.early_stopping = {"min_delta": 0.001, "patience": 10}

Construction and launch:

.. code-block:: python

    trainer = CINC2021Trainer(
        model,
        dataset_cls=CINC2021Dataset,
        model_config=ModelCfg,
        train_config=TrainCfg,
        collate_fn=collate_fn,
        device=device,
    )
    best = trainer.train()   # the best checkpoint dict; blocks until done

Provided by the base class
--------------------------

- **checkpoints**: best-on-monitor and periodic checkpoints via
  ``CkptMixin.save`` with the **full** model config (``keep_checkpoint_max``
  bounds disk usage; checkpoints reload anywhere via ``from_checkpoint``)
- **early stopping** on the monitored metric; **resume** from a checkpoint
- **criterion**: chosen by name from the train config (``TrainCfg.loss``),
  set up by the base class
- **logging**: per-run log files auto-named from the config, tensorboard
  support
- **seeding**: pass ``worker_init_fn`` in the DataLoaders (the
  benchmark shows how) and call
  :func:`~torch_ecg.utils.seed_everything` once in ``main()``; with a
  fixed seed a run is fully determined
