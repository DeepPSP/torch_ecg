.. _concepts/architecture:

Model assembly
==============

torch_ecg models are assembled bottom-up from registered building
blocks. This page explains the layering, the registries that connect
the layers, the static shape-inference contract every model honors,
and the two output conventions you will meet in the codebase.

The layering
------------

1. **Building-block configs** — :mod:`torch_ecg.model_configs` holds
   ready-made :class:`~torch_ecg.cfg.CFG` objects for every backbone
   family (``resnet``, ``mobilenet_v1/v2/v3``, ``vgg16``,
   ``densenet``, ``multi_scopic``, ``xception``, ``regnet``, plus the
   ``Transformer`` attention block and classifier/RNN heads), grouped
   so a full model config such as ``ECG_CRNN_CONFIG`` is just a few
   references to block configs.
2. **Backbones** — feature extractors registered in the
   ``BACKBONES`` registry; any of them can serve as the ``cnn`` of a
   task model without touching the task model's code.
3. **Task models** — the top level in the ``MODELS`` registry
   (:class:`~torch_ecg.models.ECG_CRNN`, ``ECG_UNET``, ...); they own
   the input/output contract (leads, signal length, classes) and
   combine backbone + attention + head into one serializable unit.

Registries
----------

Registries are the package's extension points — a name-to-class map
with a uniform ``build(name, **kwargs)`` entry point:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Registry
     - Contents
   * - ``MODELS`` / ``BACKBONES``
     - task models / feature-extraction backbones
   * - ``ATTN_LAYERS``
     - attention blocks (``Transformer``, ``SE``, ...)
   * - ``OPTIMIZERS`` / ``SCHEDULERS``
     - assembled by name from ``train_config``
   * - ``LOSSES``
     - criterion classes (assembled from ``train_config.loss``)
   * - ``PREPROCESSORS`` / ``AUGMENTERS``
     - preprocessing / online augmentation transforms

Registering a custom backbone is a decorator away
(``@BACKBONES.register()``); the
:ref:`User Guide <user_guide>` shows a worked example.

The ``compute_output_shape`` contract
-------------------------------------

Every model and backbone implements
``compute_output_shape(seq_len, batch_size=None)``, which returns the
output shape **without running a forward pass**. This makes model
composition verifiable at construction time — a task model asks its
backbone for the feature shape to size its head, and you can sanity
check a config in one line:

.. code-block:: python

   from torch_ecg.models import ECG_CRNN
   from torch_ecg.model_configs import ECG_CRNN_CONFIG

   model = ECG_CRNN(classes=["AF", "I-AVB", "LBBB", "Normal"], n_leads=12,
                    config=ECG_CRNN_CONFIG)
   model.compute_output_shape(2000)     # (None, 4)

The classification head globally pools over time, so the logits shape
is ``(batch, n_classes)`` regardless of input length.

Outputs: tensors today, dataclasses going forward
-------------------------------------------------

The classic task models (``ECG_CRNN`` and friends) return plain
logit tensors. Newer code — the
`benchmarks <https://github.com/DeepPSP/torch_ecg/tree/master/benchmarks>`__
being the reference — wraps model outputs in typed dataclasses from
:mod:`torch_ecg.components.outputs`
(:class:`~torch_ecg.components.outputs.ClassificationOutput`,
``MultiLabelClassificationOutput``,
:class:`~torch_ecg.components.outputs.WaveDelineationOutput`,
``RPeaksDetectionOutput``, ...): fields carry explicit names
(``logits``, ``bin_pred``, ``pred_classes``, ...), the objects
support ``+`` for batching and remain plain
:class:`~torch_ecg.cfg.CFG` dicts underneath, so checkpoints and
logging keep working. When writing a new task model, prefer the
dataclass contract.

System design
-------------

The overall workflow of the package — data files flow through readers,
preprocessing and augmentation into tensors, which feed the trainer
driving the model, while the config file drives model construction
and checkpoints enable resuming:

.. tikz:: Main Modules and Work Flow.
   :align: center
   :libs: arrows.meta, positioning

   \tikzstyle{block} = [rectangle, draw, text width = 6em, text centered, rounded corners, inner sep = 3pt, minimum height = 1.0em]
   \tikzstyle{arr} = [semithick, -Stealth]

   \node[block] (dr) {Data Reader};
   \node[block, right = 1.7em of dr] (pm) {Preprocessor Manager};
   \node[block, right = 2.1em of pm] (am) {Augmenter Manager};
   \node[below = 2.51em of dr] (df) {Data Files};
   \node[below = 1.8em of pm] (rd) {Raw Data};
   \node[below = 1.8em of am] (ud) {Uniformized Data};
   \node[right = 1.7em of ud] (tensor) {Tensors};

   \node[below right = 1.8em and 0.6em of tensor] (trainer) {Trainer};
   \node[below = 5em of tensor] (model) {Model};
   \node[left = 2.7em of model] (cfg) {Config File (Dictionary)};
   \node[block, right = 5.8em of am] (lm) {Logging Manager};
   \node[left = 3.5em of trainer, text width = 3.5em, text centered] (resume) {resume from};
   \node[left = 2em of resume] (ckpt) {Checkpoint};

   \node[above right = 0.2em and 1.7em of trainer] (tm) {Trained Model};
   \node[below right = 0.2em and 1.7em of trainer] (rm) {Recorded Metrics};

   \draw[arr] ([yshift=-0.2em]dr.south) -- (df);
   \draw[arr] ([yshift=-0.2em]pm.south) -- (rd);
   \draw[arr] ([yshift=-0.2em]am.south) -- (ud);
   \draw[arr] (df) -- (rd);
   \draw[arr] (rd) -- (ud);
   \draw[arr] (ud) -- (tensor);
   \draw[arr] (tensor.south) -- ([yshift=0.6em]trainer.west);
   \draw[arr] ([yshift=-0.5em]lm) -- (trainer);
   \draw[arr] (cfg) -- (model);
   \draw[arr] (model.north) -- ([yshift=-0.6em]trainer.west);
   \draw[arr] (trainer) -- (tm.west);
   \draw[arr] (trainer) -- (rm.west);
   \path[] (ckpt) edge (resume);
   \draw[arr] (resume) -- (trainer);

   \draw[black,rounded corners=20,thick] ([xshift=-0.7em, yshift=2.3em]dr.west) rectangle ([xshift=0.7em, yshift=-3.3em]rm.east);

   \draw[black,rounded corners=5, dashed, thick] ([xshift=-0.7em, yshift=1.1em]cfg.west) rectangle ([xshift=0.7em, yshift=-0.9em]model.east);
