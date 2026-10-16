""" """

from pathlib import Path

from benchmarks.train_crnn_cinc2021.train import main

_DB_DIR = Path(__file__).absolute().parents[2] / "sample-data" / "cinc2021"

_LEAD_SETS = [
    "twelve_leads",
    "six_leads",
    "four_leads",
    "three_leads",
    "two_leads",
]


def test_crnn_cinc2021_pipeline(tmp_path):
    """

    thin wrapper over the benchmarks.train_crnn_cinc2021 example,
    exercising all five Challenge lead sets on the bundled sample data.

    """
    best_model_dicts = main(
        [
            "--db-dir",
            str(_DB_DIR),
            "--lead-set",
            "all",
            "--smoke",
            "--num-workers",
            "0",
            "--outdir",
            str(tmp_path / "cinc2021"),
        ]
    )
    assert set(best_model_dicts) == set(_LEAD_SETS)
