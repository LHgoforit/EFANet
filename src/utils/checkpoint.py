import os
import torch

def save_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None, path='checkpoint.pth'):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()

    torch.save(state, path)
    print(f"[Checkpoint] Saved to {path}")

def load_checkpoint(model, optimizer=None, scheduler=None, scaler=None, path='checkpoint.pth', map_location='cpu'):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    print(f"[Checkpoint] Loaded from {path} (Epoch {epoch})")
    return epoch
