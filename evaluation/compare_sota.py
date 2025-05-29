# evaluation/compare_sota.py
# ------------------------------------------------------------------
#  Benchmark script: EFANet vs. five state-of-the-art baselines
#  Outputs a markdown table (PSNR | SSIM | LPIPS | Params | FLOPs)
#  for a given <scale, dataset> pair specified via YAML config.
#  -----------------------------------------------------------------
#  Usage:
#     python evaluation/compare_sota.py --cfg configs/efanet_x4_celeba.yaml \
#           --ckpt_dir checkpoints/
#
#  Assumptions:
#  • Each model checkpoint is named   <method>_<scale>.pth
#  • All baseline model classes expose .from_pretrained(ckpt)
#  • Dataset split text files are defined in the YAML config
# ------------------------------------------------------------------

import argparse
import yaml
from pathlib import Path
from collections import OrderedDict
import torch

from torch.utils.data import DataLoader
from src.data.face_dataset import FaceDataset
from evaluation.metrics import MetricMeter


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def instantiate_model(name: str, ckpt: Path, device: torch.device):
    """
    Dynamically import model class and load its weights.
    All baseline repos should expose XXXNet.from_pretrained().
    """
    if name.lower() == "efanet":
        from src.efanet.efanet import EFANet

        model = EFANet(**cfg["model"]).to(device)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        pkg = __import__(f"third_party.{name.lower()}", fromlist=["Net"])
        model = pkg.Net.from_pretrained(ckpt).to(device)
    model.eval()
    return model


@torch.no_grad()
def infer(model, lr_img, scale: int):
    pred = model(lr_img)
    pred = torch.clamp(pred, 0.0, 1.0)
    if pred.shape[-1] != lr_img.shape[-1] * scale:
        raise RuntimeError("Model output size mismatch.")
    return pred


# ------------------------------------------------------------------
# Core evaluation loop
# ------------------------------------------------------------------
def evaluate(cfg: dict, ckpt_dir: Path, device: torch.device):

    # Model list
    methods = ["GFPGAN", "GCFSR", "Uformer", "CTCNet", "ELSFace", "EFANet"]
    results = OrderedDict()

    # Dataset
    test_set = FaceDataset(
        root=cfg["dataset"]["root"],
        id_file=cfg["dataset"]["val_list"],
        hr_size=cfg["dataset"]["hr_size"],
        scale=cfg["experiment"]["scale"],
        is_train=False,
    )
    loader = DataLoader(
        test_set,
        batch_size=8,
        shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=True,
    )

    for m in methods:
        ckpt = ckpt_dir / f"{m.lower()}_{cfg['experiment']['scale']}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model = instantiate_model(m, ckpt, device)
        meter = MetricMeter()

        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = infer(model, lr, cfg["experiment"]["scale"])
            meter.update(sr, hr)

        psnr, ssim, lpips = meter.value()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        flops = getattr(model, "flops", lambda: float("nan"))() / 1e9

        results[m] = {
            "PSNR": f"{psnr:.3f}",
            "SSIM": f"{ssim:.4f}",
            "LPIPS": f"{lpips:.4f}",
            "Params(M)": f"{params:.1f}",
            "FLOPs(G)": f"{flops:.2f}",
        }

    # Print markdown table
    header = "| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Params (M) ↓ | FLOPs (G) ↓ |"
    line = "|---" * 6 + "|"
    print(header)
    print(line)
    for k, v in results.items():
        print(
            f"| {k} | {v['PSNR']} | {v['SSIM']} | {v['LPIPS']} | "
            f"{v['Params(M)']} | {v['FLOPs(G)']} |"
        )


# ------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--ckpt_dir", required=True, type=Path)
    parser.add_argument("--cuda", default=0, type=int)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    dev = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.cfg)
    evaluate(cfg, args.ckpt_dir, dev)
