# ECG UNET

Re-production of the model proposed in Reference \[[1](#ref1)\] using LUDB

## Training

A typical experiment of training a U-Net model for wave delineation

Curves of train loss and validation f1 score | Curves of the mean error and standard deviation
:-------------------------------------------:|:-----------------------------------------------:
<img src="images/ludb-unet-score-loss.svg" alt="score-loss" width="400"/>  |  <img src="images/ludb-unet-me-std.svg" alt="me-std" width="400"/>

## Evaluation

An example of a trained wave delineation U-Net model evaluated on a 10s segment from the validation set

<img src="images/ludb-unet-val-example.svg" alt="val-example" width="800"/>

## References

1. <a name="ref1"></a> Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
