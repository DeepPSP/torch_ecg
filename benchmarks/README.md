# benchmarks of CinC and CPSC challenges, and using other databases

a large part are migrated from other DeepPSP repositories, some are implemented in the old fasion, being inconsistent with the new system architecture of `torch_ecg`, hence need updating and testing

| Benchmark                           | Architecture              | Source                                                  | Finished           | Updated            | Tested             |
| ----------------------------------- | ------------------------- | ------------------------------------------------------- | ------------------ | ------------------ | ------------------ |
| [CinC2020](train_crnn_cinc2020/)    | CRNN                      | [DeepPSP/cinc2020](https://github.com/DeepPSP/cinc2020) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [CinC2021](train_crnn_cinc2021/)    | CRNN                      | [DeepPSP/cinc2021](https://github.com/DeepPSP/cinc2021) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [CinC2022](train_mtl_cinc2022/)[^1] | Multi Task Learning (MTL) | [DeepPSP/cinc2022](https://github.com/DeepPSP/cinc2022) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [CPSC2019](train_multi_cpsc2019/)   | SequenceTagging/U-Net     | NA                                                      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [CPSC2020](train_hybrid_cpsc2020/)  | CRNN/SequenceTagging      | [DeepPSP/cpsc2020](https://github.com/DeepPSP/cpsc2020) | :heavy_check_mark: | :x:                | :x:                |
| [CPSC2021](train_hybrid_cpsc2021/)  | CRNN/SequenceTagging/LSTM | [DeepPSP/cpsc2021](https://github.com/DeepPSP/cpsc2021) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [LUDB](train_unet_ludb/)            | U-Net                     | NA                                                      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

[^1]: Although `CinC2022` dealt with acoustic cardiac signals (phonocardiogram, PCG), the tasks and signals can be treated similarly.

## Known Issues

1. Slicing data for CPSC2021 is too slow. An offline generated (sliced) dataset is hosted at [Kaggle](https://www.kaggle.com/wenh06/cpsc2021-sliced).
2. Dataset for LUDB is too slow
3. cli for training models are completely not tested
