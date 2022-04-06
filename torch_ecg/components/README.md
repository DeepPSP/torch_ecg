# Components

This module consists of frequently used components such as loggers, trainers, etc.

## [Loggers](/torch_ecg/components/loggers.py)
Loggers including
1. CSV logger
2. text logger
3. tensorboard logger
are implemented and manipulated uniformly by a manager.

## [Outputs](/torch_ecg/components/outputs.py)
The `Output` classes implemented in this module serve as containers for ECG downstream task model outputs, including
- `ClassificationOutput`
- `MultiLabelClassificationOutput`
- `SequenceTaggingOutput`
- `WaveDelineationOutput`
- `RPeaksDetectionOutput`

each having some required fields (keys), and is able to hold an arbitrary number of custom fields. These classes are useful for the computation of metrics.

## [Metrics](/torch_ecg/components/metrics.py)
This module has the following pre-defined (built-in) `Metrics` classes:
- `ClassificationMetrics`
- `RPeaksDetectionMetrics`
- `WaveDelineationMetrics`

These metrics are computed according to either [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall), or some published literatures.

## [Trainer](/torch_ecg/components/trainer.py)
An abstract base class `BaseTrainer` is implemented, in which some common steps in building a training pipeline (workflow) are impemented. A few task specific methods are assigned as `abstractmethod`s, for example the method
```python
evaluate(self, data_loader:DataLoader) -> Dict[str, float]
```
for evaluation on the validation set during training and perhaps further for model selection and early stopping.
