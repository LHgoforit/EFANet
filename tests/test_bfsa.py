import pytest
import torch
from torch import nn
from src.efanet.bfsa import BFSA

@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("channels", [32, 64, 96])
@pytest.mark.parametrize("height, width", [(32, 32), (64, 64)])
def test_bfsa_forward_shape(batch_size, channels, height, width):
    x = torch.randn(batch_size, channels, height, width)
    bfsa = BFSA(dim=channels)

    out = bfsa(x)
    assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"

def test_bfsa_fp16_consistency():
    x_fp32 = torch.randn(2, 64, 64, 64)
    x_fp16 = x_fp32.half()

    bfsa = BFSA(dim=64).eval()
    bfsa.half()

    with torch.no_grad():
        out_fp32 = BFSA(dim=64).eval()(x_fp32).float()
        out_fp16 = bfsa(x_fp16).float()

    diff = torch.abs(out_fp32 - out_fp16).mean().item()
    assert diff < 1e-2, f"FP16 inconsistency: mean diff = {diff}"

def test_bfsa_jit_traceable():
    model = BFSA(dim=48).eval()
    dummy_input = torch.randn(1, 48, 32, 32)

    try:
        traced = torch.jit.trace(model, dummy_input)
        out = traced(dummy_input)
        assert out.shape == dummy_input.shape
    except Exception as e:
        pytest.fail(f"BFSA JIT tracing failed: {str(e)}")

def test_bfsa_gradient_flow():
    x = torch.randn(2, 48, 32, 32, requires_grad=True)
    bfsa = BFSA(dim=48)
    y = bfsa(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0), "No gradients flowed through BFSA"

