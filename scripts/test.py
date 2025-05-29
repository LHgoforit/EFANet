# src/scripts/test.py
# ------------------------------------------------------------------
# EFANet Evaluation Script for Face Super-Resolution (x4/x8/x16)
# Loads trained weights, runs inference, saves output images,
# and computes quantitative evaluation metrics.
# ------------------------------------------------------------------

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.efanet.efanet import EFANet
from src.data.face_dataset import FaceDataset
from src.data.transforms import build_transforms
from src.evaluation.metrics import compute_metrics


def parse_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained model weights')
    args = parser.parse_args()

    cfg = parse_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = EFANet(
        scale=cfg['model']['scale'],
        dim=cfg['model']['dim'],
        heads=cfg['model']['heads'],
        num_blocks=cfg['model']['num_blocks'],
        fusion=cfg['model']['fusion']
    ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Dataset
    transform = build_transforms(cfg['data']['size'])
    test_set = FaceDataset(cfg['data']['val_root'], transform=transform, scale=cfg['model']['scale'], eval_mode=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    save_root = os.path.join(cfg['test']['save_dir'], f"x{cfg['model']['scale']}")
    os.makedirs(save_root, exist_ok=True)

    psnr_total, ssim_total = 0.0, 0.0
    count = 0

    with torch.no_grad():
        for i, (lr, hr, name) in enumerate(test_loader):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)

            # Save result image
            out_path = os.path.join(save_root, f"{name[0]}.png")
            save_image(torch.clamp(sr, 0, 1), out_path)

            # Metrics
            psnr, ssim = compute_metrics(sr, hr)
            psnr_total += psnr
            ssim_total += ssim
            count += 1

    avg_psnr = psnr_total / count
    avg_ssim = ssim_total / count
    print(f"[x{cfg['model']['scale']}] Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM: {avg_ssim:.4f}")


if __name__ == '__main__':
    main()
