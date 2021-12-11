# benchmarks of CinC and CPSC challenges, and using other databases

a large part are migrated from other DeepPSP repositories, some are implemented in the old fasion, being inconsistent with the new system architecture of `torch_ecg`, hence need updating and testing

| Benchmark  | Architecture              | Source                                                  | Updated            | Tested             |
| ---------- | ------------------------- | ------------------------------------------------------- | ------------------ | ------------------ |
| CinC2020   | CRNN                      | [DeepPSP/cinc2020](https://github.com/DeepPSP/cinc2020) | :x:                | :x:                |
| CinC2021   | CRNN                      | [DeepPSP/cinc2021](https://github.com/DeepPSP/cinc2021) | :heavy_check_mark: | :x:                |
| CPSC2019   | SequenceTagging/U-Net     | NA                                                      | :x:                | :x:                |
| CPSC2020   | CRNN/SequenceTagging      | [DeepPSP/cpsc2020](https://github.com/DeepPSP/cpsc2020) | :x:                | :x:                |
| CPSC2021   | CRNN/SequenceTagging/LSTM | [DeepPSP/cpsc2021](https://github.com/DeepPSP/cpsc2021) | :x:                | :x:                |
| LUDB       | U-Net                     | NA                                                      | :x:                | :x:                |
