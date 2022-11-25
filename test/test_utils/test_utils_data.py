"""
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import torch
import pytest

from torch_ecg.utils.utils_data import (
    get_mask,
    class_weight_to_sample_weight,
    rdheader,
    ensure_lead_fmt,
    ensure_siglen,
    ECGWaveForm,
    masks_to_waveforms,
    mask_to_intervals,
    uniform,
    stratified_train_test_split,
    cls_to_bin,
    generate_weight_mask,
)


def test_get_mask():
    mask = get_mask((12, 5000), np.arange(250, 5000 - 250, 400), 50, 50)
    assert mask.shape == (12, 5000)
    assert np.unique(mask).tolist() == [0, 1]
    assert mask.sum(axis=1).tolist() == [1200] * 12
    intervals = intervals = get_mask(
        (12, 5000), np.arange(250, 5000 - 250, 400), 50, 50, return_fmt="intervals"
    )
    assert intervals == [
        [200, 300],
        [600, 700],
        [1000, 1100],
        [1400, 1500],
        [1800, 1900],
        [2200, 2300],
        [2600, 2700],
        [3000, 3100],
        [3400, 3500],
        [3800, 3900],
        [4200, 4300],
        [4600, 4700],
    ]
    for idx in range(12):
        assert intervals == mask_to_intervals(mask[idx], 1)
    assert (get_mask(5000, np.arange(250, 5000 - 250, 400), 50, 50) == mask[0]).all()


def test_mask_to_intervals():
    mask = np.zeros(100, dtype=int)
    mask[10:20] = 1
    mask[80:95] = 1
    mask[50:60] = 2
    assert mask_to_intervals(mask, vals=1) == [[10, 20], [80, 95]]
    assert mask_to_intervals(mask, vals=[0, 2]) == {
        0: [[0, 10], [20, 50], [60, 80], [95, 100]],
        2: [[50, 60]],
    }
    assert mask_to_intervals(mask) == {
        0: [[0, 10], [20, 50], [60, 80], [95, 100]],
        1: [[10, 20], [80, 95]],
        2: [[50, 60]],
    }
    assert mask_to_intervals(mask, vals=[1, 2], right_inclusive=True) == {
        1: [[10, 19], [80, 94]],
        2: [[50, 59]],
    }


def test_class_weight_to_sample_weight():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 2])
    assert np.allclose(
        class_weight_to_sample_weight(y, class_weight="balanced"),
        [
            0.25,
            0.25,
            0.25,
            0.25,
            1 / 3,
            1 / 3,
            1 / 3,
            1.0,
        ],
        atol=1e-5,
    )
    assert np.allclose(
        class_weight_to_sample_weight(y, class_weight=[1, 1, 3]),
        [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1.0],
        atol=1e-5,
    )
    assert np.allclose(
        class_weight_to_sample_weight(y, class_weight={0: 2, 1: 1, 2: 3}),
        [2 / 3, 2 / 3, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 1 / 3, 1.0],
        atol=1e-5,
    )

    y = ["NSR", "NSR", "AF", "NSR", "PVC", "AF", "NSR", "AF"]
    assert np.allclose(
        class_weight_to_sample_weight(y, class_weight={"NSR": 1, "AF": 2, "PVC": 3}),
        [1 / 3, 1 / 3, 2 / 3, 1 / 3, 1.0, 2 / 3, 1 / 3, 2 / 3],
        atol=1e-5,
    )
    assert (class_weight_to_sample_weight(y, class_weight=None) == 1).all()
    with pytest.raises(
        AssertionError,
        match="""if `y` are of type str, then class_weight should be "balanced" or a dict""",
    ):
        class_weight_to_sample_weight(y, class_weight=[1, 2, 3])


def test_rdheader():
    sample_path = Path(__file__).parents[2] / "sample-data" / "cinc2021" / "E07500.hea"
    header_lines = sample_path.read_text().splitlines()
    assert isinstance(rdheader(header_lines), wfdb.io.record.Record)
    assert (
        rdheader(str(sample_path)).__dict__
        == rdheader(header_lines).__dict__
        == wfdb.rdheader(str(sample_path).replace(".hea", "")).__dict__
    )
    with pytest.raises(
        FileNotFoundError, match="file `not_exist_file\\.hea` not found"
    ):
        rdheader("not_exist_file")
    with pytest.raises(
        TypeError, match="header_data must be str or sequence of str, but got"
    ):
        rdheader(1)


def test_ensure_lead_fmt():
    values = np.random.randn(5000, 12)
    new_values = ensure_lead_fmt(values, fmt="lead_first")
    assert new_values.shape == (12, 5000)
    assert np.allclose(new_values, values.T)
    assert np.allclose(ensure_lead_fmt(new_values, fmt="lead_last"), values)
    with pytest.raises(ValueError, match="not valid 2-lead signal"):
        ensure_lead_fmt(values, n_leads=2)
    with pytest.raises(ValueError, match="not valid fmt: `not_valid_fmt`"):
        ensure_lead_fmt(values, fmt="not_valid_fmt")


def test_ensure_siglen():
    values = np.random.randn(12, 4629)
    new_values = ensure_siglen(values, 5000, fmt="lead_first")
    assert new_values.shape == (12, 5000)
    assert np.allclose(
        new_values[:, (5000 - 4629) // 2 : (5000 - 4629) // 2 + 4629], values
    )
    new_values = ensure_siglen(values, 4000, tolerance=0.1, fmt="lead_first")
    assert new_values.shape == (math.ceil((4629 - 4000) / (0.1 * 4000)), 12, 4000)
    new_values = ensure_siglen(values, 4000, tolerance=0.2, fmt="lead_first")
    assert new_values.shape == (1, 12, 4000)

    values = np.random.randn(4629, 12)
    new_values = ensure_siglen(values, 3000, fmt="channel_last")
    assert new_values.shape == (3000, 12)
    assert np.allclose(
        new_values, values[(4629 - 3000) // 2 : (4629 - 3000) // 2 + 3000]
    )
    new_values = ensure_siglen(values, 3000, fmt="channel_last", tolerance=0.1)
    assert new_values.shape == (math.ceil((4629 - 3000) / (0.1 * 3000)), 3000, 12)


def test_masks_to_waveforms():
    class_map = {
        "pwave": 1,
        "qrs": 2,
        "twave": 3,
    }
    masks = np.zeros((2, 500), dtype=int)  # 2 leads, 5000 samples
    masks[:, 100:150] = 1
    masks[:, 160:205] = 2
    masks[:, 250:340] = 3
    waveforms = masks_to_waveforms(masks, class_map=class_map, fs=500)
    assert waveforms.keys() == {"lead_1", "lead_2"}
    assert waveforms["lead_1"] == [
        ECGWaveForm(name="pwave", onset=100, offset=150, peak=np.nan, duration=100.0),
        ECGWaveForm(name="qrs", onset=160, offset=205, peak=np.nan, duration=90.0),
        ECGWaveForm(name="twave", onset=250, offset=340, peak=np.nan, duration=180.0),
    ]

    new_waveforms = masks_to_waveforms(
        masks, class_map=class_map, fs=500, leads=["III", "aVR"]
    )
    assert new_waveforms.keys() == {"III", "aVR"}
    assert new_waveforms["III"] == waveforms["lead_1"]
    assert new_waveforms["aVR"] == waveforms["lead_2"]

    masks = np.zeros((500,), dtype=int)  # 1 lead, 5000 samples
    masks[100:150] = 1
    masks[160:205] = 2
    masks[250:340] = 3
    new_waveforms = masks_to_waveforms(masks, class_map=class_map, fs=500)
    assert new_waveforms.keys() == {"lead_1"}
    assert new_waveforms["lead_1"] == waveforms["lead_1"]

    with pytest.raises(
        ValueError, match="masks should be of dim 1 or 2, but got a 3d array"
    ):
        masks_to_waveforms(
            np.ones((2, 12, 500), dtype=int), class_map=class_map, fs=500
        )


def test_uniform():
    arr = uniform(-10, 100, 999)
    assert len(arr) == 999
    assert all([-10 <= x <= 100 for x in arr])


def test_stratified_train_test_split():
    sample_data_path = (
        Path(__file__).parents[2] / "sample-data" / "cinc2022_training_data.csv"
    )
    df_sample = pd.read_csv(sample_data_path)
    cols = ["Murmur", "Age", "Sex", "Pregnancy status", "Outcome"]
    test_ratio = 0.2
    df_train, df_test = stratified_train_test_split(
        df_sample, cols, test_ratio=test_ratio
    )
    for col in cols:
        classes = df_sample[col].apply(str).unique()
        for c in classes:
            if len(df_sample[df_sample[col].apply(str) == c]) < 20:
                continue
            assert len(df_test[df_test[col].apply(str) == c]) / len(
                df_sample[df_sample[col].apply(str) == c]
            ) == pytest.approx(test_ratio, abs=0.03)

    extra_col = "extra_col"
    df_sample.loc[df_sample.index, extra_col] = [1 for _ in range(len(df_sample))]
    df_sample.loc[int(0.5 * len(df_sample)), extra_col] = 0
    with pytest.warns(
        RuntimeWarning,
        match="invalid columns: \\['extra_col'\\], each of which has classes with only one member \\(row\\)",
    ):
        stratified_train_test_split(
            df_sample, cols + [extra_col], test_ratio=test_ratio
        )


def test_cls_to_bin():
    cls_array = torch.randint(0, 26, size=(1000,))
    bin_array = cls_to_bin(cls_array)
    assert bin_array.shape == (1000, 26)
    assert set(np.unique(bin_array)).issubset({0, 1})

    cls_array = np.random.randint(0, 26, size=(1000,))
    bin_array = cls_to_bin(cls_array)
    assert bin_array.shape == (1000, 26)
    assert set(np.unique(bin_array)).issubset({0, 1})
    with pytest.raises(
        AssertionError,
        match="num_classes must be greater than 0 and equal to the max value of `cls_array` if `cls_array` is 1D and `num_classes` is specified",
    ):
        cls_to_bin(cls_array, num_classes=25)

    cls_array = np.random.randint(0, 2, size=(1000, 26))
    bin_array = cls_to_bin(cls_array, num_classes=26)
    assert (bin_array == cls_array).all()
    with pytest.raises(
        AssertionError, match="`cls_array` should be 1D if num_classes is not specified"
    ):
        cls_to_bin(cls_array)


def test_generate_weight_mask():
    target_mask = np.zeros(50000, dtype=int)
    target_mask[500:14000] = 1
    target_mask[35800:44600] = 1
    fg_weight = 2.0
    fs = 500
    reduction = 1
    radius = 0.8
    boundary_weight = 5.0
    weight_mask = generate_weight_mask(
        target_mask, fg_weight, fs, reduction, radius, boundary_weight
    )

    assert weight_mask.shape == (50000,)
    reduction = 10
    weight_mask_reduced = generate_weight_mask(
        target_mask, fg_weight, fs, reduction, radius, boundary_weight
    )
    assert weight_mask_reduced.shape == (5000,)
    assert weight_mask_reduced.sum() * reduction == pytest.approx(
        weight_mask.sum(), rel=1e-2
    )

    with pytest.raises(AssertionError, match="`target_mask` should be 1D"):
        generate_weight_mask(
            np.zeros((100, 500)), fg_weight, fs, reduction, radius, boundary_weight
        )
    with pytest.raises(AssertionError, match="`target_mask` should be binary"):
        generate_weight_mask(
            np.full((100,), 2), fg_weight, fs, reduction, radius, boundary_weight
        )
    with pytest.raises(
        AssertionError, match="`reduction` should be a real number greater than 1"
    ):
        generate_weight_mask(target_mask, fg_weight, fs, 0.6, radius, boundary_weight)
