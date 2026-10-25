.. _examples:

Examples
========

The :code:`benchmarks/` directory of the `repository
<https://github.com/DeepPSP/torch_ecg/tree/master/benchmarks>`__ contains
eight complete, self-contained training examples -- one per challenge /
database -- written against the modern library API.  They double as the
reference usage examples for the whole package: each one shows reader and
dataset usage, model construction, training with the library trainer, and
evaluation with the official challenge metrics where applicable.

Each benchmark is runnable as a module with a :code:`--smoke` debug mode
(a few hundred iterations on a subset, no full training):

.. code-block:: bash

    python -m benchmarks.train_crnn_cinc2021.train --help
    python -m benchmarks.train_crnn_cinc2021.train --smoke

The examples
------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Benchmark
     - Task / Database
     - Architectures
   * - :code:`train_crnn_cinc2021`
     - 26-class classification, CinC Challenge 2021 (all 6 lead sets)
     - :code:`ECG_CRNN`
   * - :code:`train_crnn_cinc2020`
     - 27-class classification, CinC Challenge 2020 (strict subset of the 2021 data)
     - :code:`ECG_CRNN`
   * - :code:`train_unet_ludb`
     - ECG waveform delineation, LUDB
     - :code:`ECG_UNET`
   * - :code:`train_multi_cpsc2019`
     - In-hospital arrhythmia detection, CPSC2019
     - :code:`ECG_SEQ_LAB_NET`, :code:`ECG_SUBTRACT_UNET`, :code:`ECG_CRNN`
   * - :code:`train_hybrid_cpsc2021`
     - LOC/RLOC regression, CPSC2021 (3 sub-tasks)
     - :code:`ECG_UNET_LUDB` variants
   * - :code:`train_hybrid_cpsc2020`
     - End-to-end + segmentation challenge score, CPSC2020
     - :code:`ECG_CRNN`, :code:`ECG_SEQ_LAB_NET`
   * - :code:`train_crnn_cinc2023`
     - 24-class classification on ECG + metal-oxide gas sensors, CinC Challenge 2023
     - :code:`CRNN_CINC2023`
   * - :code:`train_mtl_cinc2022`
     - Multi-task (classification + regression) on circulatory sounds, CinC Challenge 2022
     - :code:`CRNN_CINC2022`, :code:`Wav2Vec2` variants

Highlights to look for
----------------------

- **Determinism**: every :code:`train.py` calls :func:`seed_everything`
  and uses :code:`worker_init_fn`; a given :code:`--seed` fully
  determines the run.
- **Library datasets/readers/metrics**: the examples consume
  :mod:`torch_ecg.databases` readers and datasets instead of local
  copies, and compute official challenge metrics via the library.
- **Offline preparation**: :code:`train_hybrid_cpsc2021` runs the
  (idempotent) offline preparation via
  :meth:`~torch_ecg.databases.cpsc_databases.cpsc2021.CPSC2021Dataset.persistence`
  automatically.
- **Checkpoints**: written via :meth:`~torch_ecg.utils.utils_nn.CkptMixin.save`
  with the full model config, loadable with
  :meth:`~torch_ecg.utils.utils_nn.CkptMixin.from_checkpoint`.

Tests of the examples
---------------------

:code:`test/test_pipelines/` contains thin pytest wrappers over the
benchmarks (fast lanes run the non-downloading parts; the nightly full
suite trains briefly on real data), so the examples are continuously
verified against the library.
