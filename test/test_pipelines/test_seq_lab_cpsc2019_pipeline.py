""" """

from pathlib import Path

from benchmarks.train_multi_cpsc2019.train import main

_DB_DIR = Path(__file__).absolute().parents[2] / "sample-data" / "cpsc2019"

_ARCHS = [
    "seq_lab_crnn",
    "unet",
    "subtract_unet",
]


def test_seq_lab_cpsc2019_pipeline(tmp_path):
    """

    thin wrapper over the benchmarks.train_multi_cpsc2019 example,
    exercising all three architectures on the bundled sample data.

    """
    for arch in _ARCHS:
        main(
            [
                "--db-dir",
                str(_DB_DIR),
                "--arch",
                arch,
                "--smoke",
                "--num-workers",
                "0",
                "--outdir",
                str(tmp_path / arch),
            ]
        )
