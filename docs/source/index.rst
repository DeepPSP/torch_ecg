.. torch_ecg documentation master file, created by
   sphinx-quickstart on Mon Jul  5 15:53:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to torch-ecg's documentation!
=====================================

ECG Deep Learning Framework Implemented using PyTorch.

The system design is depicted as follows

.. tikz:: Main Modules and Work Flow.
   :align: center
   :libs: shapes,arrows.meta,decorations.pathmorphing,backgrounds,positioning,fit,petri,calc,hobby

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

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   install
   tutorial

.. toctree::
   :glob:
   :caption: API Reference
   :maxdepth: 1

   databases
   models
   augmenters
   preprocessors
   components
   utils

.. toctree::
   :caption: Examples
   :maxdepth: 1

   examples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
