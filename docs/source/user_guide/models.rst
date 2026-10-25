.. _ug_models:

Models and configurations
=========================

Task models
-----------

The task-level models take a config, the number of leads and the number
of classes/leads-out:

.. code-block:: python

    from torch_ecg.models import ECG_CRNN
    from torch_ecg.model_configs import ECG_CRNN_CONFIG

    config = ECG_CRNN_CONFIG.copy()
    config.cnn.name = "resnet_nature_comm_bottle_neck_se"
    model = ECG_CRNN(config, n_leads=12, n_classes=26)

    import torch
    out = model(torch.randn(2, 12, 4000))       # (2, 26)
    model.compute_output_shape(bs=2, input_len=4000)   # (2, 26) -- shape is computable
    # without running a forward pass

The full catalogue of task models (classification, sequence labeling,
U-Net segmentation, RR-LSTM, ...) is in :mod:`torch_ecg.models`; each
supports ``compute_output_shape`` so that architectures can be composed
and checked statically.

Building blocks
---------------

Models are assembled from registered backbones and blocks; the ready
made block configs are catalogued in :mod:`torch_ecg.model_configs`
(VGG / ResNet families / TResNet / DenseNet / Xception / MobileNet /
MultiScopic CNN / attention modules / RNN / MLP heads).  Custom backbones
plug in through the registries:

.. code-block:: python

    from torch_ecg.models import BACKBONES

    @BACKBONES.register()
    class MyBackbone(nn.Module):
        ...

    config.cnn.name = "MyBackbone"   # then usable by name in any config

Checkpoints
-----------

Checkpoints written via
:meth:`~torch_ecg.utils.utils_nn.CkptMixin.save` carry the **full**
model config, so a trained model is one line to reload, on any machine:

.. code-block:: python

    model.save("ckpt_dir", train_config=..., model_config=config)

    from torch_ecg.models import ECG_CRNN
    model, train_config = ECG_CRNN.from_checkpoint("ckpt_dir")
    preds = model.inference(sig)     # database-agnostic inference helpers

Remote (pre-trained) checkpoints work the same way via
``from_remote(url, model_dir)``.  See the ``test_seq_lab_net`` /
``test_mixin_classes`` tests for round-trips of all three on-disk
formats (single-file safetensors, directory, ``.pth``).
