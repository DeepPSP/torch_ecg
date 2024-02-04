"""
"""

import itertools
import platform
import shutil
import warnings
from pathlib import Path

import packaging.version
import pytest

from torch_ecg.databases import CINC2021
from torch_ecg.utils._ecg_plot import create_signal_dictionary, ecg_plot, inches_to_dots, leadNames_12

warnings.simplefilter(action="ignore", category=DeprecationWarning)


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "sample-data" / "cinc2021"
_TMP_DIR = Path(__file__).parents[1] / "tmp" / "test_ecg_plot"
_TMP_DIR.mkdir(exist_ok=True, parents=True)
###############################################################################

# clean up the tmp dir
for f in _TMP_DIR.iterdir():
    if f.is_file():
        f.unlink()
    elif f.is_dir():
        shutil.rmtree(f)

reader = CINC2021(_CWD)


def test_ecg_plot():
    # test inches_to_dots
    assert inches_to_dots(1.2, 200) == 240

    rec = "HR06000"
    signal = reader.load_data(rec)
    leads = reader.all_leads
    sample_rate = reader.get_fs(rec)

    signal_dict = create_signal_dictionary(signal[:, : int(2.5 * sample_rate)], leads)
    assert isinstance(signal_dict, dict)
    assert len(signal_dict) == len(leads)
    assert set(signal_dict.keys()) == set(leads)

    signal_dict["fullII"] = signal_dict["II"].copy()

    grid = itertools.product(
        [2, 4],  # columns
        [None, leadNames_12],  # lead_index
        ["None", "II"],  # full_mode
        ["bw", "color"],  # style
        [0, 2],  # standard_colours
    )

    # skip if Python version < 3.8
    # matplotlib (version 3.5.3) in Python 3.7.10 has a bug that will raise an error
    # RuntimeError: Cannot get window extent w/o renderer
    # when calling `ax.get_window_extent()` in `ecg_plot()`
    if packaging.version.parse(platform.python_version()) < packaging.version.parse("3.8"):
        return

    x_grid_dots, y_grid_dots = ecg_plot(
        ecg={}, sample_rate=100, columns=4, rec_file_name=reader.get_absolute_path(rec), output_dir=_TMP_DIR
    )
    assert (x_grid_dots, y_grid_dots) == (0, 0)

    for columns, lead_index, full_mode, style, standard_colours in grid:
        x_grid_dots, y_grid_dots = ecg_plot(
            ecg=signal_dict,
            sample_rate=sample_rate,
            columns=columns,
            rec_file_name=reader.get_absolute_path(rec),
            output_dir=_TMP_DIR,
            resolution=100,
            pad_inches=2,
            lead_index=lead_index,
            full_mode=full_mode,
            store_text_bbox=True,
            title="Test ECG Plot",
            style=style,
            show_lead_name=True,
            show_grid=True,
            show_dc_pulse=True,
            standard_colours=standard_colours,
            bbox=True,
            save_format=None,
        )
        # NOTE: setting store_text_bbox=False and bbox=True will raise an error
        # assert no image file is saved in the output_dir
        assert len([f for f in _TMP_DIR.iterdir() if f.is_file()]) == 0

    signal_dict = create_signal_dictionary(signal[:, : int(10 * sample_rate)], leads)
    signal_dict["fullII"] = signal_dict["II"].copy()

    for fmt in ["png", "pdf", "svg", "jpg"]:
        x_grid_dots, y_grid_dots = ecg_plot(
            ecg=signal_dict,
            sample_rate=sample_rate,
            columns=1,
            rec_file_name=reader.get_absolute_path(rec),
            output_dir=_TMP_DIR,
            resolution=100,
            pad_inches=2,
            lead_index=leadNames_12,
            full_mode="II",
            store_text_bbox=True,
            papersize="A3",
            title="Test ECG Plot",
            style="bw",
            show_lead_name=True,
            show_grid=True,
            show_dc_pulse=False,
            standard_colours=True,
            bbox=True,
            save_format=fmt,
        )
        # assert file is saved in the output_dir
        output_image_file = _TMP_DIR / reader.get_absolute_path(rec).with_suffix(f".{fmt}").name
        assert output_image_file.exists()

    with pytest.raises(AssertionError, match="Total time of ECG signal of one row cannot exceed 10 seconds"):
        ecg_plot(
            ecg=signal_dict,
            sample_rate=sample_rate,
            columns=4,
            rec_file_name=reader.get_absolute_path(rec),
            output_dir=_TMP_DIR,
        )

    with pytest.raises(AssertionError, match="save_format must be one of"):
        ecg_plot(
            ecg=signal_dict,
            sample_rate=sample_rate,
            columns=1,
            rec_file_name=reader.get_absolute_path(rec),
            output_dir=_TMP_DIR,
            save_format="abc",
        )
