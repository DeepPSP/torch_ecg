.. _user_guide:

User Guide
==========

This guide follows the typical usage flow: reading a database,
wrapping it in a dataset, selecting and configuring a model, training
with the library trainer, and evaluating with official metrics.

.. toctree::
   :maxdepth: 1

   data
   preprocess
   models
   training
   evaluation

Every code block in this guide is either executed as part of the test
suite or taken from the :ref:`examples <examples>` (the `benchmarks/
<https://github.com/DeepPSP/torch_ecg/tree/master/benchmarks>`__
directory), which double as continuously-tested reference usage.
