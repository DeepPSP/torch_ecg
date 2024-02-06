# [PhysioNet/CinC Challenge 2021](https://moody-challenge.physionet.org/2021/)

Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021

![docker-ci](https://github.com/DeepPSP/cinc2021/actions/workflows/docker-image.yml/badge.svg)

## Digest of Top Solutions (ranked by [final challenge score](https://docs.google.com/spreadsheets/d/1cTLRmSLS1_TOwx-XnY-QVoUyO2rFyPUGTHRzNm3u8EM/edit?usp=sharing))

1. [ISIBrno-AIMT](https://www.cinc.org/2021/Program/accepted/14_Preprint.pdf): Custom ResNet + MultiHeadAttention + Custom Loss
2. [DSAIL_SNU](https://www.cinc.org/2021/Program/accepted/80_Preprint.pdf): SE-ResNet + Custom Loss (from Asymmetric Loss)
3. [NIMA](https://www.cinc.org/2021/Program/accepted/352_Preprint.pdf): Time-Freq Domain + Custom CNN
4. [cardiochallenger](https://www.cinc.org/2021/Program/accepted/234_Preprint.pdf): Inception-ResNet + Channel Self-Attention + Custom Loss
5. [USST_Med](https://www.cinc.org/2021/Program/accepted/105_Preprint.pdf): SE-ResNet + Focal Loss + Data Re-labeling Model
6. [CeZIS](https://www.cinc.org/2021/Program/accepted/78_Preprint.pdf): ResNet50 + FlowMixup
7. [SMS+1](https://www.cinc.org/2021/Program/accepted/24_Preprint.pdf): Custom CNN + Hand-crafted Features + Asymmetric Loss
8. [DataLA_NUS](https://www.cinc.org/2021/Program/accepted/122_Preprint.pdf): EfficientNet + SE-ResNet + Custom Loss

Other teams that are not among official entries, but among [unofficial entries](https://docs.google.com/spreadsheets/d/1iMKPXDvqfyQlwhsd4N6CjKZccikhsIkSDygLEsICqsw/edit?usp=sharing):

1. [HeartBeats](https://www.cinc.org/2021/Program/accepted/63_Preprint.pdf): SE-ResNet + Sign Loss + Model Ensemble

`Aizip-ECG-team` and `Proton` had high score on the hidden test set, but [did not submitted papers](https://docs.google.com/spreadsheets/d/1sSKA9jMp8oT2VqyX4CTirIT3m5lSohIuk5GWf-Cq8FU/edit?usp=sharing), hence not described here.

## Conference Website and Conference Programme

[Website](http://www.cinc2021.org/), [Programme](https://cinc.org/archives/2021/), [IEEE Xplore](https://ieeexplore.ieee.org/xpl/conhome/9662654/proceeding), [Poster](images/CinC2021_poster.pdf)

## Data Preparation

One can download training data from [GCP](https://console.cloud.google.com/storage/browser/physionetchallenge2021-public-datasets),
and use `python prepare_dataset -i {data_directory} -v` to prepare the data for training

## Deep Models

Deep learning models are constructed using [torch_ecg](https://github.com/DeepPSP/torch_ecg), which has already been added as a submodule.

## Final Results

Final results are on the [leaderboard page of the challenge official website](https://physionetchallenges.org/2021/leaderboard/) or one can find in the [offical_results folder](official_results/).

## Citation

```latex
@inproceedings{wen_cinc2021,
      title = {{Hybrid Arrhythmia Detection on Varying-Dimensional Electrocardiography: Combining Deep Neural Networks and Clinical Rules}},
     author = {Hao Wen and Jingsu Kang},
  booktitle = {{2021 Computing in Cardiology (CinC)}},
        doi = {10.23919/cinc53138.2021.9662801},
       year = {2021},
      month = {9},
  publisher = {{IEEE}},
}
@article{Kang_2022_cinc2021_iop,
     author = {Jingsu Kang and Hao Wen},
      title = {{A Study on Several Critical Problems on Arrhythmia Detection using Varying-Dimensional Electrocardiography}},
    journal = {Physiological Measurement},
        doi = {10.1088/1361-6579/ac6aa3},
       year = {2022},
      month = {4},
  publisher = {{IOP} Publishing}
}
```

## [Original Repository](https://github.com/DeepPSP/cinc2021)
