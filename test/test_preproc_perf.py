"""
Performance and Numerical Consistency Benchmarking for Preprocessors.
"""

import time

import numpy as np
import pytest
import torch

from torch_ecg.preprocessors import BandPass, BaselineRemove

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_test_signal(batch_size=32, n_leads=12, seq_len=5000, fs=500):
    # Sinusoidal signal + baseline drift + noise
    t = torch.linspace(0, seq_len / fs, seq_len)
    # Pure signal (10 Hz)
    sig = torch.sin(2 * np.pi * 10 * t)
    # Baseline drift (0.1 Hz)
    drift = 2.0 * torch.sin(2 * np.pi * 0.1 * t)
    # Noise
    noise = 0.5 * torch.randn(seq_len)

    base_sig = sig + drift + noise
    # Expand to batch and leads
    batch_sig = base_sig.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_leads, 1)
    return batch_sig


def test_bandpass_consistency():
    fs = 500
    lowcut, highcut = 0.5, 45
    sig_torch = generate_test_signal(batch_size=4, seq_len=10000, fs=fs).to(DEVICE)
    sig_np = sig_torch.cpu().numpy()

    # 1. SciPy version (via legacy path)
    bp = BandPass(fs=fs, lowcut=lowcut, highcut=highcut)
    out_np = bp._forward_numpy(sig_np)

    # 2. Torch version
    out_torch = bp._forward_torch(sig_torch)

    # Check consistency (FFT filter and IIR filter won't be identical, but should be close in passband)
    # We use MSE as a loose metric because frequency domain zeroing != time domain recursion
    diff = np.abs(out_np - out_torch.cpu().numpy())
    mse = np.mean(diff**2)
    print(f"\nBandPass Consistency (MSE): {mse:.6f}")
    # FFT filtering is usually "cleaner" than IIR, so some difference is expected
    assert mse < 0.1


def test_baseline_remove_consistency():
    fs = 500
    window1, window2 = 0.2, 0.6
    sig_torch = generate_test_signal(batch_size=4, seq_len=10000, fs=fs).to(DEVICE)
    sig_np = sig_torch.cpu().numpy()

    br = BaselineRemove(fs=fs, window1=window1, window2=window2)

    # 1. SciPy version (Median Filter)
    out_np = br._forward_numpy(sig_np)

    # 2. Torch version (Sliding Average)
    out_torch = br._forward_torch(sig_torch)

    diff = np.abs(out_np - out_torch.cpu().numpy())
    mse = np.mean(diff**2)
    print(f"\nBaselineRemove Consistency (MSE): {mse:.6f}")
    # Median filter and Sliding average are different algorithms, but should achieve same goal
    assert mse < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU performance test requires CUDA")
def test_preproc_performance():
    batch_size = 128
    n_leads = 12
    seq_len = 5000
    fs = 500
    sig_torch = generate_test_signal(batch_size, n_leads, seq_len, fs).to(DEVICE)

    bp = BandPass(fs=fs, lowcut=0.5, highcut=45).to(DEVICE)
    br = BaselineRemove(fs=fs).to(DEVICE)

    # Warmup
    _ = bp(sig_torch)
    _ = br(sig_torch)
    torch.cuda.synchronize()

    # 1. Performance of Torch path
    start = time.time()
    for _ in range(10):
        _ = bp._forward_torch(sig_torch)
        _ = br._forward_torch(sig_torch)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 10

    # 2. Performance of NumPy path (includes data transfer simulation)
    start = time.time()
    for _ in range(10):
        # Simulation of old way: move to CPU -> proc -> move back
        tmp = sig_torch.cpu().numpy()
        _ = bp._forward_numpy(tmp)
        _ = br._forward_numpy(tmp)
        _ = torch.as_tensor(tmp).to(DEVICE)
    numpy_time = (time.time() - start) / 10

    speedup = numpy_time / torch_time
    print(f"\nPerformance Result (Batch size {batch_size}):")
    print(f"NumPy Path (with transfers): {numpy_time:.4f}s")
    print(f"Torch Path (on GPU): {torch_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")

    assert speedup > 2.0


if __name__ == "__main__":
    pytest.main([__file__])
