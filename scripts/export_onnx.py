# src/scripts/export_onnx.py
# ------------------------------------------------------------------
# Exports a trained EFANet model to ONNX format for deployment.
# Supports arbitrary super-resolution scale (e.g., x4, x8, x16).
# ------------------------------------------------------------------

import os
import argparse
import torch
import yaml

from src.efanet.efanet import EFANet


def parse_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def export_onnx(cfg, weights_path, output_path, input_size=(1, 3, 64, 64)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EFANet(
        scale=cfg['model']['scale'],
        dim=cfg['model']['dim'],
        heads=cfg['model']['heads'],
        num_blocks=cfg['model']['num_blocks'],
        fusion=cfg['model']['fusion']
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'scaled_height', 3: 'scaled_width'}
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=True
    )

    print(f"ONNX model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--output', type=str, required=True, help='Output ONNX file path')
    parser.add_argument('--height', type=int, default=64, help='Input LR image height')
    parser.add_argument('--width', type=int, default=64, help='Input LR image width')
    args = parser.parse_args()

    cfg = parse_config(args.config)
    input_size = (1, 3, args.height, args.width)

    export_onnx(cfg, args.weights, args.output, input_size=input_size)


if __name__ == '__main__':
    main()
