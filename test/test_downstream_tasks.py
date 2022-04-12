"""
"""

import time
from typing import NoReturn

import torch

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from torch_ecg.model_configs import ATI_CNN_CONFIG  # noqa: F401; noqa: F401
from torch_ecg.model_configs import ECG_CRNN_CONFIG  # noqa: F401
from torch_ecg.model_configs import ECG_SEQ_LAB_NET_CONFIG  # noqa: F401
from torch_ecg.model_configs import ECG_SUBTRACT_UNET_CONFIG  # noqa: F401
from torch_ecg.model_configs import ECG_UNET_VANILLA_CONFIG  # noqa: F401
from torch_ecg.model_configs import ECG_YOLO_CONFIG  # noqa: F401
from torch_ecg.model_configs import RR_AF_CRF_CONFIG  # noqa: F401
from torch_ecg.model_configs import RR_AF_VANILLA_CONFIG  # noqa: F401
from torch_ecg.model_configs import RR_LSTM_CONFIG  # noqa: F401
from torch_ecg.models.ecg_crnn import ECG_CRNN  # noqa: F401
from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET  # noqa: F401
from torch_ecg.models.rr_lstm import RR_LSTM  # noqa: F401
from torch_ecg.models.unets.ecg_subtract_unet import ECG_SUBTRACT_UNET  # noqa: F401
from torch_ecg.models.unets.ecg_unet import ECG_UNET  # noqa: F401

_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

_IN_CHANNELS = 12
_BATCH_SIZE = 2
_SIG_LEN = 4000
_RR_LEN = 100
_TEST_EXAMPLE = torch.rand((_BATCH_SIZE, _IN_CHANNELS, _SIG_LEN)).to(_DEVICE)
_TEST_RR_EXAMPLE = torch.rand(
    (
        _RR_LEN,
        _BATCH_SIZE,
        1,
    )
).to(_DEVICE)
_TEST_CLF_CLASSES = [
    "nsr",
    "af",
    "pvc",
]
_TEST_DELI_CLASSES = [
    "qrs",
    "p",
    "t",
]


@torch.no_grad()
def test_tasks() -> NoReturn:
    """ """
    start = time.time()
    print("\n" + " Test downstream task configs ".center(80, "#") + "\n")
    # test crnn configs
    print("\n" + " Test ECG_CRNN configs ".center(50, "-") + "\n")
    ECG_CRNN.__DEBUG__ = False
    try:
        test_model = ECG_CRNN(
            classes=_TEST_CLF_CLASSES, n_leads=_IN_CHANNELS, config=ECG_CRNN_CONFIG
        ).to(_DEVICE)
        test_model.eval()
        test_output = test_model(_TEST_EXAMPLE)
        print(f"ECG_CRNN output shape = {test_output.shape}")
        del test_model
        del test_output
    except Exception as e:
        print("ECG_CRNN_CONFIG raises errors\n")
        raise e

    # test seq_lab configs
    print("\n" + " Test ECG_SEQ_LAB_NET configs ".center(50, "-") + "\n")
    ECG_SEQ_LAB_NET.__DEBUG__ = False
    try:
        test_model = ECG_SEQ_LAB_NET(
            classes=_TEST_DELI_CLASSES,
            n_leads=_IN_CHANNELS,
            config=ECG_SEQ_LAB_NET_CONFIG,
        ).to(_DEVICE)
        test_model.eval()
        test_output = test_model(_TEST_EXAMPLE)
        print(f"ECG_SEQ_LAB_NET output shape = {test_output.shape}")
        del test_model
        del test_output
    except Exception as e:
        print("ECG_SEQ_LAB_NET raises errors\n")
        raise e

    # test unet configs
    print("\n" + " Test ECG_UNET configs ".center(50, "-") + "\n")
    ECG_UNET.__DEBUG__ = False
    try:
        test_model = ECG_UNET(
            classes=_TEST_DELI_CLASSES,
            n_leads=_IN_CHANNELS,
            config=ECG_UNET_VANILLA_CONFIG,
        ).to(_DEVICE)
        test_model.eval()
        test_output = test_model(_TEST_EXAMPLE)
        print(f"ECG_UNET output shape = {test_output.shape}")
        del test_model
        del test_output
    except Exception as e:
        print("ECG_UNET_VANILLA_CONFIG raises errors\n")
        raise e

    # test subtract_unet configs
    print("\n" + " Test ECG_SUBTRACT_UNET configs ".center(50, "-") + "\n")
    ECG_SUBTRACT_UNET.__DEBUG__ = False
    try:
        test_model = ECG_SUBTRACT_UNET(
            classes=_TEST_DELI_CLASSES,
            n_leads=_IN_CHANNELS,
            config=ECG_SUBTRACT_UNET_CONFIG,
        ).to(_DEVICE)
        test_model.eval()
        test_output = test_model(_TEST_EXAMPLE)
        print(f"ECG_SUBTRACT_UNET output shape = {test_output.shape}")
        del test_model
        del test_output
    except Exception as e:
        print("ECG_SUBTRACT_UNET_CONFIG raises errors\n")
        raise e

    # test rr_lstm configs
    print("\n" + " Test RR_LSTM configs ".center(50, "-") + "\n")
    RR_LSTM.__DEBUG__ = False
    for cfg in [
        "RR_AF_CRF_CONFIG",
        "RR_AF_VANILLA_CONFIG",
        "RR_LSTM_CONFIG",
    ]:
        try:
            test_model = eval(
                f"RR_LSTM(classes=_TEST_CLF_CLASSES, config={cfg}).to(_DEVICE)"
            )
            test_model.eval()
            test_output = test_model(_TEST_RR_EXAMPLE)
            print(f"{cfg} output shape = {test_output.shape}")
            del test_model
            del test_output
        except Exception as e:
            print(f"{cfg} raises errors\n")
            raise e

    print(f"total time cost: {time.time()-start:.2f} seconds")
    print("\n" + " Finish testing downstream task configs ".center(80, "#") + "\n")


if __name__ == "__main__":
    test_tasks()
