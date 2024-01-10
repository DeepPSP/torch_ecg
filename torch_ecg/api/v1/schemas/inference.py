from typing import List, Optional

from pydantic import BaseModel


class InferenceInput(BaseModel):
    data: List[List[List[float]]]  # batch_size, n_leads, signal_length
    metadata: Optional[dict] = None


class InferenceOutput(BaseModel):
    predictions: List[List[float]]  # batch_size, n_classes
    predicted_classes: List[str]  # batch_size
