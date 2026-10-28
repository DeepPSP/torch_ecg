.. _concepts/training_internals:

Inside the trainer
==================

:mod:`torch_ecg.components.trainer` provides the training loop every
benchmark builds on. Understanding its anatomy — and its determinism
machinery — saves hours of debugging.

Anatomy of the training loop
----------------------------

``BaseTrainer`` subclasses implement ``_setup_dataloaders``,
``run_one_step`` and ``evaluate``; the base class owns the loop. Per
training step::

    out_tensors = self.run_one_step(*data)   # (preds, labels), no loss
    loss = self.criterion(*out_tensors)      # applied by the base loop

The criterion is **assembled by name** from ``train_config.loss``
through the ``LOSSES`` registry (as are the optimizer and scheduler
from their config entries), so a benchmark switches losses by editing
one config field, never code. An optional ``flooding_level`` wraps the
loss with the flooding regularizer before ``backward()``.

After each epoch, the trainer validates and then walks this decision
path:

- If ``train_config.monitor`` is set (an evaluation-metric name, e.g.
  ``"challenge_metric"``), a new best value promotes the just-saved
  checkpoint to *best model*; ``early_stopping`` (with ``patience``
  and ``min_delta``) can stop the run when no improvement accumulates
  for enough epochs.
- ``keep_checkpoint_max`` bounds how many per-epoch checkpoints stay
  on disk — the oldest are removed beyond the limit.
- If no monitor is set, the **last** model is kept as the best (the
  trainer logs this explicitly), and resuming
  (``train(..., resume=True)``) restores the model, optimizer state
  and epoch counter from the checkpoint written by this same loop.

Determinism: seeding and workers
--------------------------------

:func:`~torch_ecg.utils.seed_everything` seeds **all** generators the
package and its dependencies draw from:

- ``DEFAULTS.RNG`` — the package's canonical
  :class:`numpy.random.Generator` (recreated fresh),
- the global ``random`` and ``numpy.random`` states (third-party code
  draws from them),
- torch's CPU and CUDA generators.

With ``num_workers > 0`` there is a second, subtler issue.
PyTorch reseeds *its own* generators in every DataLoader worker
(``base_seed + worker_id``), but it knows nothing about user-held
:class:`numpy.random.Generator` instances. Under ``fork`` (the Linux
default), every worker inherits a **copy** of the parent's
``DEFAULTS.RNG``, so all workers would run identical augmentation
streams — the same "random" noise in every worker.
:func:`~torch_ecg.utils.worker_init_fn` derives a distinct,
reproducible seed from the worker-specific ``torch.initial_seed()``
and reseeds ``DEFAULTS.RNG`` per worker. Together the two functions
give bit-wise reproducible runs:

.. code-block:: python

   from torch.utils.data import DataLoader
   from torch_ecg.utils import seed_everything, worker_init_fn

   seed_everything(42)
   dl = DataLoader(ds, batch_size=32, num_workers=4,
                   worker_init_fn=worker_init_fn)

Where augmentation happens
--------------------------

Online augmentation runs **inside the DataLoader workers**: the
dataset's ``__getitem__`` applies its
:class:`~torch_ecg.augmenters.AugmenterManager` to each sample before
collation, so the training loop never sees unaugmented data and the
CPU work parallelizes across workers.

The manager's callable takes **two** arguments — ``(sig, label)`` —
because correct ECG augmentation must be able to keep the annotation
in sync with the signal. Today's classification labels are multi-hot
vectors that are invariant to signal-level transforms (nothing to
re-permute for, say, baseline wander), but transforms that *do*
reorder or reshape the signal would break if the label were not in
the signature; the two-argument contract is what keeps waveform-level
tasks (e.g. lead-sensitive augmenters) composable without API churn.
