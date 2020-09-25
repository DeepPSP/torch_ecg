Notice: this repo is no more updated. 
- Extraction of expert features can be found at https://github.com/hsd1503/ENCASE/tree/master/code/featrues_*.py
- Recent PyTorch implementation of a collection of deep models on 1d signal data can be found at https://github.com/hsd1503/resnet1d 

# ENCASE

ENCASE combines deep neural networks and expert features together for AF Classification from a short single lead ECG recording. It won the First Place in the PhysioNet/Computing in Cardiology Challenge 2017 (https://physionet.org/challenge/2017), with an overall F1 score of 0.83. The original code can be downloaded from https://physionet.org/challenge/2017/sources/shenda-hong-221.zip

Detailed description of ENCASE can be found at http://www.cinc.org/archives/2017/pdf/178-245.pdf. 
If you find the idea useful or use this code in your own work, please cite our paper as
```
@inproceedings{hong2017encase,
  author    = {Shenda Hong and Meng Wu and Yuxi Zhou and Qingyun Wang and Junyuan Shang and Hongyan Li and Junqing Xie},
  title     = {{ENCASE:} an ENsemble ClASsifiEr for {ECG} Classification Using Expert
               Features and Deep Neural Networks},
  booktitle = {CinC},
  year      = {2017},
  url       = {https://doi.org/10.22489/CinC.2017.178-245},
  doi       = {10.22489/CinC.2017.178-245}
}
```
and
```
@article{hong2019combining,
	doi = {10.1088/1361-6579/ab15a2},
	url = {https://doi.org/10.1088%2F1361-6579%2Fab15a2},
	year = 2019,
	month = {Jun},
	publisher = {{IOP} Publishing},
	volume = {40},
	number = {5},
	pages = {054009},
	author = {Shenda Hong and Yuxi Zhou and Meng Wu and Junyuan Shang and Qingyun Wang and Hongyan Li and Junqing Xie},
	title = {Combining deep neural networks and engineered features for cardiac arrhythmia detection from {ECG} recordings},
	journal = {Physiological Measurement}
}
```


## Task Description

Please refer to the Challenge website https://physionet.org/challenge/2017/#introduction and Challenge description paper http://www.cinc.org/archives/2017/pdf/065-469.pdf. 

## Dataset

**Data** Training data can be found at https://archive.physionet.org/challenge/2017/#challenge-data

**Label** Please use Revised labels (v3) at https://archive.physionet.org/challenge/2017/REFERENCE-v3.csv

**Preprocessed** Or you can download my preprocessed dataset challenge2017.pkl from https://drive.google.com/drive/folders/1AuPxvGoyUbKcVaFmeyt3xsqj6ucWZezf
