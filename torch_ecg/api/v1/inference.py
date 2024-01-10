from typing import Literal

import numpy as np
import torch
from fastapi import APIRouter, Path

from torch_ecg_volta.api.config import settings
from torch_ecg_volta.api.v1.schemas.inference import InferenceInput, InferenceOutput
from torch_ecg_volta.models import ECG_CRNN, ECG_SEQ_LAB_NET, RR_LSTM

router = APIRouter()

ModelType = Literal["ECG_CRNN", "ECG_SEQ_LAB_NET", "RR_LSTM"]
MODEL_MAP = {
    "ECG_CRNN": ECG_CRNN,
    "ECG_SEQ_LAB_NET": ECG_SEQ_LAB_NET,
    "RR_LSTM": RR_LSTM,
}


@router.post("/models/{model_type}/{version}/infer")
def infer(
    inference_input: InferenceInput,
    model_type: ModelType,
    version: int,
) -> InferenceOutput:
    # TODO: Implement model loading, inference, caching, etc.
    # TODO: Implement a batching mechanism across requests to reduce load.
    model, config = MODEL_MAP[model_type].from_checkpoint(
        settings.CHECKPOINT_DIR / model_type / str(version), device=settings.DEVICE
    )
    classes = np.array(config.classes)
    predictions = model(torch.as_tensor(inference_input.data, device=settings.DEVICE))
    predicted_classes = classes[predictions.argmax(dim=1)].tolist()
    if isinstance(predicted_classes, str):
        predicted_classes = [predicted_classes]
    return InferenceOutput(
        predictions=predictions.tolist(),
        predicted_classes=predicted_classes,
    )
