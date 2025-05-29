import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR


def build_optimizer(model, cfg):
    """
    Construct optimizer according to cfg['optimizer'].
    Supported: Adam, AdamW
    """
    opt_cfg = cfg['optimizer']
    name = opt_cfg['type'].lower()
    params = model.parameters()

    if name == 'adam':
        optimizer = optim.Adam(
            params,
            lr=opt_cfg.get('lr', 1e-4),
            betas=opt_cfg.get('betas', (0.9, 0.999)),
            eps=opt_cfg.get('eps', 1e-8),
            weight_decay=opt_cfg.get('weight_decay', 0)
        )
    elif name == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=opt_cfg.get('lr', 1e-4),
            betas=opt_cfg.get('betas', (0.9, 0.999)),
            eps=opt_cfg.get('eps', 1e-8),
            weight_decay=opt_cfg.get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"[Optimizer] Unsupported optimizer type: {name}")

    return optimizer


def build_scheduler(optimizer, cfg):
    """
    Construct LR scheduler according to cfg['scheduler'].
    Supported: StepLR, MultiStepLR, CosineAnnealingLR
    """
    sch_cfg = cfg['scheduler']
    name = sch_cfg['type'].lower()

    if name == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=sch_cfg.get('step_size', 20),
            gamma=sch_cfg.get('gamma', 0.5)
        )
    elif name == 'multistep':
        scheduler = MultiStepLR(
            optimizer,
            milestones=sch_cfg.get('milestones', [30, 60, 90]),
            gamma=sch_cfg.get('gamma', 0.5)
        )
    elif name == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg.get('T_max', 50),
            eta_min=sch_cfg.get('eta_min', 1e-6)
        )
    else:
        raise ValueError(f"[Scheduler] Unsupported scheduler type: {name}")

    return scheduler
