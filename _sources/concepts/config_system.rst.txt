.. _concepts/config_system:

The configuration system
========================

All components of torch_ecg are configured through dictionary-like
objects. This page explains the three layers of configuration, the
:class:`~torch_ecg.cfg.CFG` class that carries them, the
:data:`~torch_ecg.cfg.DEFAULTS` object that stores package-wide
settings, and a checkpoint regression that shows why the distinction
between the layers matters in practice.

Three layers of configuration
-----------------------------

A trained model is described by three nested layers:

1. **Training configuration** (``train_config``) — owned by the
   trainer: optimizer, learning-rate scheduler, loss, monitor metric,
   early stopping, batch size, and everything about *how* to train.
2. **Outer model configuration** (``model_config``) — the full config
   a task model needs to be rebuilt from scratch, including fields the
   architecture itself never reads, such as ``classes`` and
   ``n_leads``.
3. **Inner architecture configuration** (``self.config`` inside the
   model) — what the model actually consumes: which backbone, its
   building-block parameters, attention and classifier settings.

The outer layer is what :meth:`~torch_ecg.models.ECG_CRNN.from_checkpoint`
persists and reloads. Losing the distinction between the outer and
inner layers is not a cosmetic issue — see the case study below.

``CFG``: one object, two views
------------------------------

:class:`~torch_ecg.cfg.CFG` subclasses ``dict`` and mirrors every
entry as an attribute, so both ``config["key"]`` and ``config.key``
work and stay in sync:

.. code-block:: python

   from torch_ecg.cfg import CFG

   c = CFG(hehe={"a": 1, "b": 2})
   c.update(hehe={"a": [-1], "c": -10})
   c["hehe"]          # {'a': [-1], 'b': 2, 'c': -10}
   c.hehe.pop("a")    # disappears from the dict view too

Note the ``update`` semantics: it merges **recursively** — existing
sub-dictionaries are updated key by key instead of being replaced, and
a plain ``dict`` value overwriting a non-dict entry is deep-copied so
the two configs do not share mutable values. Keys that would shadow
the ``dict`` API (``copy``, ``get``, ``update``, ...) are rejected
with an error rather than silently breaking the container.

``copy()`` versus ``deepcopy()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``CFG`` inherits ``dict.copy``, which returns a **plain ``dict``**:
the attribute view is gone, and so is recursive isolation. Always use
:func:`copy.deepcopy` to duplicate a config:

.. code-block:: python

   from copy import deepcopy
   from torch_ecg.cfg import CFG

   tc = CFG(classes=["AF", "I-AVB"], n_leads=12)
   tc.copy()           # plain dict — `tc_copy.classes` raises AttributeError
   deepcopy(tc)        # still a CFG, fully isolated from `tc`

``DEFAULTS``: package-wide settings
-----------------------------------

:data:`~torch_ecg.cfg.DEFAULTS` is a module-level ``CFG`` holding the
package defaults:

- **dtype** — a :class:`~torch_ecg.cfg.DTYPE` dataclass keeping the
  numpy/torch/string/`wfdb` integer views of one dtype consistent
  (``DEFAULTS.DTYPE.TORCH``, ``DEFAULTS.str_dtype``, ...);
  :func:`~torch_ecg.cfg.DEFAULTS.change_dtype` switches it globally.
- **RNG** — ``DEFAULTS.RNG`` is the canonical
  :class:`numpy.random.Generator` of the package, a *single* global
  stream that every stochastic component draws from.
  :func:`~torch_ecg.utils.seed_everything` reseeds it (plus the global
  ``random`` / ``numpy.random`` states and the torch CPU/CUDA
  generators), which provides run-to-run reproducibility (see
  :ref:`concepts/training_internals`).
- **cache paths** — ``log_dir``, ``checkpoints``, ``model_dir``,
  ``working_dir`` under ``~/.cache/torch_ecg``.

``DEFAULTS.device`` is **lazy**: it is not probed at import time, but
detected and cached on first access. Probing the CUDA driver during
``import torch_ecg`` adds overhead in every Python 3.14+ multiprocessing
worker (the forkserver/spawn start methods re-import the package) and
would freeze the device before ``CUDA_VISIBLE_DEVICES`` could be set.

Case study: the 0.0.32 checkpoint regression
--------------------------------------------

Version 0.0.32 moved checkpoint saving to safetensors, and with it
started storing only the model's *inner* ``self.config``. The outer
fields — ``classes``, ``n_leads``, and anything else the model's
``__init__`` reads before touching the architecture — were silently
dropped. ``from_checkpoint`` reconstructs the model via
``cls(config=...)``; for any model whose ``__init__`` reads
``config.classes``, loading a 0.0.32 checkpoint crashed.

Version 0.0.34 fixes the asymmetry at the saving side:
:meth:`CkptMixin.save <torch_ecg.utils.utils_nn.CkptMixin.save>`
gained a ``model_config`` parameter and the trainer passes its full
``model_config``, so trainer-written checkpoints again contain the
complete outer config, as the pre-0.0.32 ``torch.save`` path did.

The second half of the fix generalizes beyond this case: serializing a config
requires :func:`~torch_ecg.utils.utils_nn.make_safe_globals` to strip
values that cannot round-trip (callables, path-like values, ...).
Before the fix these fields **vanished silently** (a checkpoint lost
its ``db_dir`` with no trace); now the function emits a
``RuntimeWarning`` listing every dropped field. The underlying pattern,
converting a silent failure into an explicit warning, is applied
throughout the package.
