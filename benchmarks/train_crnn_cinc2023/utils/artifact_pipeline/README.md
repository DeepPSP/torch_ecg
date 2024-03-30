# Artifact pipeline

This folder is modified from <https://github.com/bdsp-core/icare-dl/blob/main/Artifact_pipeline.zip>

The following artifacts are detected in the pipeline:

- NaN in EEG
- overly high/low amplitude,
- flat signal,
- NaN in feature,
- NaN in spectrum,
- overly high/low total power,
- muscle artifact,
- multiple assessment scores,
- spurious spectrum,
- fast rising decreasing,
- 1Hz artifact

The main file is [segment_EEG.py](segment_EEG.py). It takes EEG signals (typically preprocessed) and outputs a list of segments with labels.
The segments are 10s long and the labels are as above plus a normal label.

Based on this artifact detection pipeline, we build functions generating signal quality index (SQI) for longer time windows (e.g. 5 minutes)
The SQI is defined as the percentage of segments with normal label in the time window. The functions are in [utils/sqi.py](../sqi.py).
