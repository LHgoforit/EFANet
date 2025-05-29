import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint
from utils.metrics_io import update_metrics

class Trainer:
    def __init__(self, model, dataloader, criterion, optimizer, scheduler, cfg, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device

        self.logger = get_logger(cfg['log_dir'])
        self.scaler = GradScaler(enabled=cfg.get('use_amp', True))
        self.epoch = 0
        self.best_psnr = -1

    def train(self):
        self.logger.info("Starting training loop...")
        for epoch in range(self.cfg['epochs']):
            self.epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.time()

            for i, batch in enumerate(self.dataloader['train']):
                lr = batch['lr'].to(self.device)
                hr = batch['hr'].to(self.device)

                self.optimizer.zero_grad()
                with autocast(enabled=self.cfg.get('use_amp', True)):
                    sr = self.model(lr)
                    loss = self.criterion(sr, hr)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

            self.scheduler.step()
            avg_loss = epoch_loss / len(self.dataloader['train'])
            self.logger.info(f"Epoch [{epoch+1}/{self.cfg['epochs']}], Loss: {avg_loss:.4f}, Time: {time.time() - epoch_start:.2f}s")

            if (epoch + 1) % self.cfg['val_interval'] == 0:
                psnr = self.validate()
                if psnr > self.best_psnr:
                    self.best_psnr = psnr
                    save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, psnr, self.cfg['ckpt_dir'], is_best=True)

    def validate(self):
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0

        with torch.no_grad():
            for batch in self.dataloader['val']:
                lr = batch['lr'].to(self.device)
                hr = batch['hr'].to(self.device)

                with autocast(enabled=self.cfg.get('use_amp', True)):
                    sr = self.model(lr)

                psnr, ssim = update_metrics(sr, hr)
                total_psnr += psnr
                total_ssim += ssim

        avg_psnr = total_psnr / len(self.dataloader['val'])
        avg_ssim = total_ssim / len(self.dataloader['val'])
        self.logger.info(f"Validation @ Epoch {self.epoch+1}: PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
        return avg_psnr
