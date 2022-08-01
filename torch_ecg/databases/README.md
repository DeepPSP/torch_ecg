# database_reader
python modules to facilitate the reading of various databaese from [PhysioNet](https://physionet.org/), [CPSC](http://www.icbeb.org/#), [NSRR](https://sleepdata.org/), etc.

Migrated and improved from [DeepPSP/database_reader](https://github.com/DeepPSP/database_reader)

After migration, all should be tested again, the progression:

| Database      | Source                                                           | Implemented        | Fully Tested[^1]   | Has `Dataset`      |
| ------------- | ---------------------------------------------------------------- | ------------------ | ------------------ | ------------------ |
| AFDB          | [PhysioNet](https://physionet.org/content/afdb/1.0.0/)           | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| ApneaECG      | [PhysioNet](https://physionet.org/content/apnea-ecg/1.0.0/)      | :heavy_check_mark: | :x:                | :x:                |
| CinC2017      | [PhysioNet](https://physionet.org/content/challenge-2017/1.0.0/) | :heavy_check_mark: | :x:                | :x:                |
| CinC2018      | [PhysioNet](https://physionet.org/content/challenge-2018/1.0.0/) | :x:                | :x:                | :x:                |
| CinC2020      | [PhysioNet](https://physionet.org/content/challenge-2020/1.0.1/) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CinC2021      | [PhysioNet](https://physionet.org/content/challenge-2021/1.0.2/) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| LTAFDB        | [PhysioNet](https://physionet.org/content/ltafdb/1.0.0/)         | :heavy_check_mark: | :x:                | :x:                |
| LUDB          | [PhysioNet](https://physionet.org/content/ludb/1.0.1/)           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| MITDB         | [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QTDB          | [PhysioNet](https://physionet.org/content/qtdb/1.0.0/)           | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| SHHS          | [NSRR](https://sleepdata.org/datasets/shhs)                      | :heavy_check_mark: | :x:                | :x:                |
| CPSC2018      | [CPSC](http://2018.icbeb.org/Challenge.html)                     | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| CPSC2019      | [CPSC](http://2019.icbeb.org/Challenge.html)                     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CPSC2020      | [CPSC](http://2020.icbeb.org/CSPC2020)                           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CPSC2021[^2]  | [CPSC](http://2021.icbeb.org/CPSC2021)                           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| SPH           | [Figshare](https://doi.org/10.6084/m9.figshare.c.5779802.v1)     | :heavy_check_mark: | :heavy_check_mark: | :x:                |

[^1]: Since the classes are migrated from [DeepPSP/database_reader](https://github.com/DeepPSP/database_reader), some are not tested for newly added features.
[^2]: The dataset `CPSC2021` is also hosted at [PhysioNet](https://www.physionet.org/content/cpsc2021/1.0.0/).


## Basic Usage
```python
>>> from torch_ecg.databases import CINC2021
>>> dr = CINC2021("/path/to/the/directory/of/CINC2021-data/)
converting dtypes of columns `diagnosis` and `diagnosis_scored`...
>>> len(dr)
88253
>>> dr.load_data(0, leads=["I", "II"], data_format="channel_last", units="uv")
array([[ 28.,   7.],
       [ 39.,  11.],
       [ 45.,  15.],
       ...,
       [258., 248.],
       [259., 249.],
       [259., 250.]], dtype=float32)
>>> dr.load_ann(0)
{'rec_name': 'A0001',
 'nb_leads': 12,
 'fs': 500,
 'nb_samples': 7500,
 'datetime': datetime.datetime(2020, 5, 12, 12, 33, 59),
 'age': 74,
 'sex': 'Male',
 'medical_prescription': 'Unknown',
 'history': 'Unknown',
 'symptom_or_surgery': 'Unknown',
 'diagnosis': {'diagnosis_code': ['59118001'],
  'diagnosis_abbr': ['RBBB'],
  'diagnosis_fullname': ['right bundle branch block']},
 'diagnosis_scored': {'diagnosis_code': ['59118001'],
  'diagnosis_abbr': ['RBBB'],
  'diagnosis_fullname': ['right bundle branch block']},
 'df_leads':       filename fmt  byte_offset  ...  checksum block_size  lead_name
 I    A0001.mat  16           24  ...     -1716          0          I
 II   A0001.mat  16           24  ...      2029          0         II
 III  A0001.mat  16           24  ...      3745          0        III
 aVR  A0001.mat  16           24  ...      3680          0        aVR
 aVL  A0001.mat  16           24  ...     -2664          0        aVL
 aVF  A0001.mat  16           24  ...     -1499          0        aVF
 V1   A0001.mat  16           24  ...       390          0         V1
 V2   A0001.mat  16           24  ...       157          0         V2
 V3   A0001.mat  16           24  ...     -2555          0         V3
 V4   A0001.mat  16           24  ...        49          0         V4
 V5   A0001.mat  16           24  ...      -321          0         V5
 V6   A0001.mat  16           24  ...     -3112          0         V6
 
 [12 rows x 12 columns]}
>>> dr.get_labels(30000, scored_only=True, fmt="f")  # full names
['sinus arrhythmia',
 'right axis deviation',
 'incomplete right bundle branch block']
>>> dr.get_labels(30000, scored_only=True, fmt="a")  # abbreviations
['SA', 'RAD', 'IRBBB']
>>> dr.get_labels(30000, scored_only=False, fmt="s")  # SNOMED CT Code
['427393009', '445211001', '47665007', '713426002']
```

## Functionalities

Each `Database` has the following basic functionalities
1. Download from data archive (mainly PhysioNet) using the `download` method
```python
>>> from torch_ecg.databases import MITDB
>>> dr = MITDB(db_dir="/any/path/even/if/does/not/exists/")
>>> # download the compressed zip file of MITDB
>>> # and extract to `dr.db_dir`
>>> dr.download(compressed=True)
```
2. Loading data and annotations using `load_data` and `load_ann` respectively (ref. [Basic Usage](#basic-usage)).
3. `plot` functions.

For a `PhysioNetDataBase`, one has the `helper` function for looking up annotation meanings
```python
>>> from torch_ecg.databases import MITDB
>>> dr = MITDB(db_dir="/any/path/even/if/does/not/exists/")
>>> dr.helper("beat")
MIT-BIH Arrhythmia Database
--- helpler - beat ---
{   '/': 'Paced beat',
    '?': 'Beat not classified during learning',
    'A': 'Atrial premature beat',
    'B': 'Bundle branch block beat (unspecified)',
    'E': 'Ventricular escape beat',
    'F': 'Fusion of ventricular and normal beat',
    'J': 'Nodal (junctional) premature beat',
    'L': 'Left bundle branch block beat',
    'N': 'Normal beat',
    'Q': 'Unclassifiable beat',
    'R': 'Right bundle branch block beat',
    'S': 'Supraventricular premature or ectopic beat (atrial or nodal)',
    'V': 'Premature ventricular contraction',
    'a': 'Aberrated atrial premature beat',
    'e': 'Atrial escape beat',
    'f': 'Fusion of paced and normal beat',
    'j': 'Nodal (junctional) escape beat',
    'n': 'Supraventricular escape beat (atrial or nodal)',
    'r': 'R-on-T premature ventricular contraction'}
>>> dr.helper("rhythm")
MIT-BIH Arrhythmia Database
--- helpler - rhythm ---
{   '(AB': 'Atrial bigeminy',
    '(AFIB': 'Atrial fibrillation',
    '(AFL': 'Atrial flutter',
    '(B': 'Ventricular bigeminy',
    '(BII': '2° heart block',
    '(IVR': 'Idioventricular rhythm',
    '(N': 'Normal sinus rhythm',
    '(NOD': 'Nodal (A-V junctional) rhythm',
    '(P': 'Paced rhythm',
    '(PREX': 'Pre-excitation (WPW)',
    '(SBR': 'Sinus bradycardia',
    '(SVTA': 'Supraventricular tachyarrhythmia',
    '(T': 'Ventricular trigeminy',
    '(VFL': 'Ventricular flutter',
    '(VT': 'Ventricular tachycardia'}
```

## TODO
1. use the attribute `_df_records` to maintain paths, etc. uniformly
