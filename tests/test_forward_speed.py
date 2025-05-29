import pytest
import torch
import time
from src.efanet.efanet import EFANet

@pytest.mark.parametrize("scale", [4, 8, 16])
@pytest.mark.parametrize("height, width", [(64, 64), (128, 128), (256, 256)])
def test_forward_speed(scale, height, width):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EFANet(scale=scale).to(device)
    model.eval()

    input_tensor = torch.randn(1, 3, height, width).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Timing
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / 50 * 1000
    print(f"[Scale x{scale}] Input: {height}x{width} → Avg inference: {avg_time_ms:.2f} ms")

    # Sanity check on output
    with torch.no_grad():
        output = model(input_tensor)
    expected_height = height * scale
    expected_width = width * scale
    assert output.shape[2] == expected_height
    assert output.shape[3] == expected_width
