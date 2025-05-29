# src/scripts/benchmark_latency.py
# ------------------------------------------------------------------
# Benchmarks EFANet runtime latency and throughput for a given input size.
# Measures average inference time, warm-up time, and FPS under GPU or CPU.
# ------------------------------------------------------------------

import argparse
import time
import torch

from src.efanet.efanet import EFANet


def benchmark(model, input_tensor, warmup=20, repeat=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

        start = time.time()
        for _ in range(repeat):
            _ = model(input_tensor)
        end = time.time()

    avg_latency = (end - start) / repeat * 1000  # in milliseconds
    fps = 1000.0 / avg_latency
    return avg_latency, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, required=True, help='SR scale (e.g., 4, 8, 16)')
    parser.add_argument('--height', type=int, default=64, help='Input height (LR)')
    parser.add_argument('--width', type=int, default=64, help='Input width (LR)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to benchmark on')
    parser.add_argument('--repeat', type=int, default=100, help='Number of repetitions')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = EFANet(scale=args.scale).to(device)

    input_tensor = torch.randn(1, 3, args.height, args.width).to(device)
    avg_latency, fps = benchmark(model, input_tensor, repeat=args.repeat)

    print(f"Benchmarking EFANet x{args.scale} on {args.device.upper()}")
    print(f"Input Size : {args.height}x{args.width}")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"Throughput : {fps:.2f} FPS")


if __name__ == '__main__':
    main()
