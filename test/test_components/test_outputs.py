"""
"""

import numpy as np
import pytest

from torch_ecg.cfg import DEFAULTS
from torch_ecg.components.metrics import ClassificationMetrics, RPeaksDetectionMetrics, WaveDelineationMetrics
from torch_ecg.components.outputs import (
    BaseOutput,
    ClassificationOutput,
    MultiLabelClassificationOutput,
    RPeaksDetectionOutput,
    SequenceLabellingOutput,
    SequenceTaggingOutput,
    WaveDelineationOutput,
)


class TestClassificationOutput:
    classes = ["AF", "NSR", "SPB"]
    batch_size = 32
    num_classes = len(classes)

    def test_classification_output(self):
        prob = DEFAULTS.RNG.random((self.batch_size, self.num_classes))
        pred = (prob == prob.max(axis=1, keepdims=True)).astype(int)
        output = ClassificationOutput(classes=self.classes, pred=pred, prob=prob)
        with pytest.raises(
            AssertionError,
            match="`labels` or `label` must be stored in the output for computing metrics",
        ):
            output.compute_metrics()

        output.label = DEFAULTS.RNG_randint(0, self.num_classes - 1, (self.batch_size,))
        metrics = output.compute_metrics()
        assert isinstance(metrics, ClassificationMetrics)

        prob = DEFAULTS.RNG.random((self.batch_size, self.num_classes))
        pred = np.argmax(prob, axis=1)
        output = ClassificationOutput(classes=self.classes, pred=pred, prob=prob)
        output.label = DEFAULTS.RNG_randint(0, self.num_classes - 1, (self.batch_size,))
        metrics = output.compute_metrics()
        assert isinstance(metrics, ClassificationMetrics)

    def test_append(self):
        prob = DEFAULTS.RNG.random((self.batch_size, self.num_classes))
        pred = (prob == prob.max(axis=1, keepdims=True)).astype(int)
        output = ClassificationOutput(classes=self.classes, pred=pred, prob=prob)

        prob = DEFAULTS.RNG.random((self.batch_size, self.num_classes))
        pred = (prob == prob.max(axis=1, keepdims=True)).astype(int)
        output_1 = ClassificationOutput(classes=self.classes, pred=pred, prob=prob)

        prob = DEFAULTS.RNG.random((self.batch_size, self.num_classes))
        pred = (prob == prob.max(axis=1, keepdims=True)).astype(int)
        output_2 = ClassificationOutput(classes=self.classes, pred=pred, prob=prob)

        prob = DEFAULTS.RNG.random((1, self.num_classes))
        pred = (prob == prob.max(axis=1, keepdims=True)).astype(int)
        output_3 = ClassificationOutput(classes=self.classes, pred=pred, prob=prob)

        output.append(output_1)
        assert output.pred.shape == (self.batch_size * 2, self.num_classes)
        assert output.prob.shape == (self.batch_size * 2, self.num_classes)

        output.append([output_2, output_3])
        assert output.pred.shape == (self.batch_size * 3 + 1, self.num_classes)
        assert output.prob.shape == (self.batch_size * 3 + 1, self.num_classes)

        with pytest.raises(AssertionError, match="`values` must be of the same type as `self`"):
            output_4 = MultiLabelClassificationOutput(
                classes=self.classes,
                pred=np.ones((self.batch_size, self.num_classes)),
                prob=np.ones((self.batch_size, self.num_classes)),
                thr=0.5,
            )
            output.append(output_4)

        with pytest.raises(
            AssertionError,
            match="the field of ordered sequence `classes` must be the identical",
        ):
            output_5 = ClassificationOutput(
                classes=self.classes[::-1],
                pred=np.ones((self.batch_size, self.num_classes)),
                prob=np.ones((self.batch_size, self.num_classes)),
            )
            output.append(output_5)


class TestMultiLabelClassificationOutput:
    classes = ["AF", "NSR", "SPB"]
    batch_size = 32
    num_classes = len(classes)
    thr = 0.5

    def test_multilabel_classification_output(self):
        prob = DEFAULTS.RNG.random((self.batch_size, self.num_classes))
        pred = (prob > self.thr).astype(int)
        output = MultiLabelClassificationOutput(classes=self.classes, thr=self.thr, pred=pred, prob=prob)
        with pytest.raises(
            AssertionError,
            match="`labels` or `label` must be stored in the output for computing metrics",
        ):
            output.compute_metrics()

        output.label = DEFAULTS.RNG_randint(0, 1, (self.batch_size, self.num_classes))
        metrics = output.compute_metrics()
        assert isinstance(metrics, ClassificationMetrics)


class TestSequenceTaggingOutput:
    classes = ["AF", "NSR", "SPB"]
    batch_size = 32
    signal_length = 5000
    num_classes = len(classes)

    def test_sequence_tagging_output(self):
        prob = DEFAULTS.RNG.random((self.batch_size, self.signal_length, self.num_classes))
        pred = (prob == prob.max(axis=2, keepdims=True)).astype(int)
        output = SequenceTaggingOutput(classes=self.classes, pred=pred, prob=prob)
        with pytest.raises(
            AssertionError,
            match="`labels` or `label` must be stored in the output for computing metrics",
        ):
            output.compute_metrics()

        tmp = DEFAULTS.RNG.random((self.batch_size, self.signal_length, self.num_classes))
        output.label = (tmp == tmp.max(axis=2, keepdims=True)).astype(int)
        metrics = output.compute_metrics()
        assert isinstance(metrics, ClassificationMetrics)


class TestSequenceLabellingOutput:
    classes = ["AF", "NSR", "SPB"]
    batch_size = 32
    signal_length = 5000
    num_classes = len(classes)

    def test_sequence_labelling_output(self):
        prob = DEFAULTS.RNG.random((self.batch_size, self.signal_length, self.num_classes))
        pred = (prob == prob.max(axis=2, keepdims=True)).astype(int)
        output = SequenceLabellingOutput(classes=self.classes, pred=pred, prob=prob)
        with pytest.raises(
            AssertionError,
            match="`labels` or `label` must be stored in the output for computing metrics",
        ):
            output.compute_metrics()

        tmp = DEFAULTS.RNG.random((self.batch_size, self.signal_length, self.num_classes))
        output.label = (tmp == tmp.max(axis=2, keepdims=True)).astype(int)
        metrics = output.compute_metrics()
        assert isinstance(metrics, ClassificationMetrics)


class TestWaveDelineationOutput:
    classes = ["N", "P", "Q"]
    batch_size = 32
    signal_length = 500
    num_leads = 12
    num_classes = len(classes)

    def test_wave_delineation_output(self):
        truth_masks = DEFAULTS.RNG_randint(
            0,
            self.num_classes - 1,
            (self.batch_size, self.num_leads, self.signal_length),
        )
        pred_probs = DEFAULTS.RNG_randint(0, 1, (self.batch_size, self.signal_length, self.num_classes))
        pred_masks = np.argmax(pred_probs, axis=2)
        pred_masks = np.repeat(pred_masks[:, np.newaxis, :], self.num_leads, axis=1)
        output = WaveDelineationOutput(classes=self.classes, mask=pred_masks, prob=pred_probs)
        class_map = {
            "pwave": 1,
            "qrs": 2,
            "twave": 3,
        }
        with pytest.raises(
            AssertionError,
            match="`labels` or `label` must be stored in the output for computing metrics",
        ):
            output.compute_metrics(class_map=class_map, fs=500)

        output.label = truth_masks
        metrics = output.compute_metrics(class_map=class_map, fs=500)
        assert isinstance(metrics, WaveDelineationMetrics)


class TestRPeaksDetectionOutput:
    batch_size = 2
    signal_length = 1000

    def test_rpeaks_detection_output(self):
        output = RPeaksDetectionOutput(
            rpeak_indices=[np.array([200, 700]), np.array([500])],
            thr=0.5,
            prob=DEFAULTS.RNG.random((self.batch_size, self.signal_length)),
        )
        with pytest.raises(
            AssertionError,
            match="`labels` or `label` must be stored in the output for computing metrics",
        ):
            output.compute_metrics(fs=500)

        output.label = [np.array([205, 700]), np.array([500, 900])]
        metrics = output.compute_metrics(fs=500)
        assert isinstance(metrics, RPeaksDetectionMetrics)


def test_base_output():
    with pytest.raises(NotImplementedError, match="Subclass must implement method `required_fields`"):
        BaseOutput()
