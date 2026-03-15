"""
Unit tests for the standardized Backbone API.
"""

from copy import deepcopy

import pytest
import torch

from torch_ecg.model_configs import ECG_CRNN_CONFIG
from torch_ecg.models.registry import BACKBONES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract all valid backbone configurations from the central config
BACKBONE_CONFIGS = []
for config_key, config_val in ECG_CRNN_CONFIG.cnn.items():
    if not isinstance(config_val, dict):
        continue

    # 1. Try to get name from the config dict
    # 2. If not in dict, use the key if it's in the registry
    # 3. If key contains a known registry name (e.g. resnet_nature_comm), extract the base name
    backbone_name = config_val.get("name")
    if backbone_name is None:
        if config_key in BACKBONES:
            backbone_name = config_key
        else:
            # Check if config_key contains any registered name as a prefix
            for registered_name in BACKBONES.list_all():
                if config_key.startswith(registered_name):
                    backbone_name = registered_name
                    break

    if backbone_name:
        BACKBONE_CONFIGS.append((backbone_name, config_key, config_val))


@pytest.mark.parametrize("backbone_name, config_key, config", BACKBONE_CONFIGS)
def test_backbone_api(backbone_name, config_key, config):
    n_leads = 12
    batch_size = 2
    seq_len = 2000

    # Skip models that are not implemented yet to avoid noisy test failures
    # These will be implemented in Phase 1.5
    try:
        model = BACKBONES.build(backbone_name, in_channels=n_leads, **deepcopy(config)).to(DEVICE)
    except NotImplementedError:
        pytest.skip(f"Backbone {backbone_name} (config: {config_key}) is not implemented yet.")
    except Exception as e:
        pytest.fail(f"Failed to build backbone {backbone_name} with config {config_key}: {e}")

    model.eval()
    inp = torch.randn(batch_size, n_leads, seq_len).to(DEVICE)

    # 1. Test forward_features existence
    assert hasattr(model, "forward_features"), f"Backbone {backbone_name} missing forward_features"

    # 2. Test forward_features output shape
    features = model.forward_features(inp)
    assert features.ndim == 3, f"Backbone {backbone_name} forward_features should return 3D tensor, got {features.ndim}D"
    assert features.shape[0] == batch_size

    # 3. Test compute_features_output_shape consistency
    # All Backbones in torch_ecg follow (seq_len, batch_size) signature
    expected_shape = model.compute_features_output_shape(seq_len, batch_size)
    assert (
        features.shape[1] == expected_shape[1]
    ), f"Backbone {backbone_name} feature channels mismatch: {features.shape[1]} vs {expected_shape[1]}"
    if expected_shape[2] is not None:
        assert (
            features.shape[2] == expected_shape[2]
        ), f"Backbone {backbone_name} feature seq_len mismatch: {features.shape[2]} vs {expected_shape[2]}"

    # 4. Test forward consistency (if model is pure feature extractor)
    out = model(inp)
    assert torch.allclose(out, features), f"Backbone {backbone_name} forward and forward_features results differ"


if __name__ == "__main__":
    pytest.main([__file__])
