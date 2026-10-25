Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`torch_ecg` requires **Python 3.10 – 3.15** and is available through pip:

.. code-block:: bash

    python -m pip install torch-ecg

The package has no extras on PyPI -- everything needed at runtime is
installed as a dependency.  For development (tests, docs, benchmarks),
install from a checkout with the :code:`dev` extra:

.. code-block:: bash

    git clone https://github.com/DeepPSP/torch_ecg.git
    cd torch_ecg
    python -m pip install -e ".[dev]"

or use pip directly via:

.. code-block:: bash

    python -m pip install "torch-ecg @ git+https://github.com/DeepPSP/torch_ecg.git"

.. note::

    :code:`torch_ecg` downloads PhysioNet databases from the
    :code:`physionet-open` AWS S3 bucket by default.  Install the `AWS
    CLI <https://docs.aws.amazon/cli/latest/userguide/getting-started-install.html>`__
    for the fastest download path; without it, a pure-Python
    :code:`boto3` backend is used (same source, somewhat slower).
