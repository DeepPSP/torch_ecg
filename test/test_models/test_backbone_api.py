"""
Unit tests for the standardized Backbone API.
"""

from copy import deepcopy

import pytest
import torch

from torch_ecg.model_configs import ECG_CRNN_CONFIG
from torch_ecg.models.registry import BACKBONES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("backbone_name", BACKBONES.list_all())
def test_backbone_api(backbone_name):
    # Skip aliases to avoid redundant tests
    if backbone_name != backbone_name.lower():
        pytest.skip(f"Skipping alias backbone name: {backbone_name}")

    n_leads = 12
    batch_size = 2
    seq_len = 2000

    # Get default config if available in ECG_CRNN_CONFIG
    config = None
    for k, v in ECG_CRNN_CONFIG.cnn.items():
        if k.lower() == backbone_name.lower():
            config = deepcopy(v)
            break

    if config is None:
        # Some backbones might not be in ECG_CRNN_CONFIG, skip for now
        # or provide a minimal dummy config if known
        pytest.skip(f"No default config found for backbone: {backbone_name}")

    try:
        model = BACKBONES.build(backbone_name, in_channels=n_leads, **config).to(DEVICE)
    except Exception as e:
        pytest.fail(f"Failed to build backbone {backbone_name} with config {config}: {e}")

    model.eval()
    inp = torch.randn(batch_size, n_leads, seq_len).to(DEVICE)

    # 1. Test forward_features existence
    assert hasattr(model, "forward_features"), f"Backbone {backbone_name} missing forward_features"

    # 2. Test forward_features output shape
    features = model.forward_features(inp)
    assert features.ndim == 3, f"Backbone {backbone_name} forward_features should return 3D tensor, got {features.ndim}D"
    assert features.shape[0] == batch_size

    # 3. Test compute_features_output_shape consistency
    expected_shape = model.compute_features_output_shape(seq_len, batch_size)
    assert (
        features.shape[1] == expected_shape[1]
    ), f"Backbone {backbone_name} feature channels mismatch: {features.shape[1]} vs {expected_shape[1]}"
    if expected_shape[2] is not None:
        assert (
            features.shape[2] == expected_shape[2]
        ), f"Backbone {backbone_name} feature seq_len mismatch: {features.shape[2]} vs {expected_shape[2]}"

    # 4. Test forward consistency (if model is pure feature extractor)
    # For now, all current backbones in torch_ecg are pure feature extractors
    out = model(inp)
    assert torch.allclose(out, features), f"Backbone {backbone_name} forward and forward_features results differ"


if __name__ == "__main__":
    pytest.main([__file__])
