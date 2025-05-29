# src/scripts/train.py
# ------------------------------------------------------------------
# EFANet Training Script for Face Super-Resolution (x4/x8/x16)
# High-quality pipeline for training EFANet on CelebA, Helen, FFHQ.
# ------------------------------------------------------------------

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.efanet.efanet import EFANet
from src.loss.charbonnier import CharbonnierLoss
from src.loss.cw_percep import CWPerceptualLoss
from src.loss.attn_align import AttentionAlignmentLoss
from src.data.face_dataset import FaceDataset
from src.data.transforms import build_transforms
from src.utils.trainer import Trainer
from src.utils.optim import build_optimizer, build_scheduler


def parse_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    cfg = parse_config(args.config)

    os.makedirs(cfg['train']['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = EFANet(
        scale=cfg['model']['scale'],
        dim=cfg['model']['dim'],
        heads=cfg['model']['heads'],
        num_blocks=cfg['model']['num_blocks'],
        fusion=cfg['model']['fusion']
    ).to(device)

    # Loss
    pixel_criterion = CharbonnierLoss()
    perceptual_criterion = CWPerceptualLoss(layer_weights=cfg['loss']['vgg_weights']).to(device)
    attention_criterion = AttentionAlignmentLoss().to(device)

    # Dataset
    transform = build_transforms(cfg['data']['size'])
    train_set = FaceDataset(cfg['data']['train_root'], transform=transform, scale=cfg['model']['scale'])
    val_set = FaceDataset(cfg['data']['val_root'], transform=transform, scale=cfg['model']['scale'], eval_mode=True)

    train_loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Optimizer & Scheduler
    optimizer = build_optimizer(cfg['optim'], model.parameters())
    scheduler = build_scheduler(cfg['sched'], optimizer)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        pixel_loss=pixel_criterion,
        perceptual_loss=perceptual_criterion,
        align_loss=attention_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=cfg['train']['save_dir'],
        eval_interval=cfg['train']['eval_interval'],
        max_epochs=cfg['train']['epochs'],
        checkpoint_interval=cfg['train']['ckpt_interval']
    )

    trainer.fit()


if __name__ == '__main__':
    main()
