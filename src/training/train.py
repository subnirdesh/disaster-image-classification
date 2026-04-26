"""
train.py — Training loop for all three models.
Usage: called from Colab notebook, not run directly.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

from src.data.dataset import build_dataloaders
from src.models.models import build_model


def get_device():
    if torch.cuda.is_available():    return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False, ncols=70):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


class EarlyStopping:
    def __init__(self, patience, path):
        self.patience = patience
        self.path = path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            torch.save(model.state_dict(), self.path)
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_model(model_name, cfg, drive_dir, labels_csv, num_classes):
    print(f"\n{'='*50}\nTraining: {model_name.upper()}\n{'='*50}")

    device = get_device()
    print(f"Device: {device}")

    ckpt_path = f"{drive_dir}/checkpoints/{model_name}/best.pth"
    log_path  = f"{drive_dir}/logs/{model_name}/training_log.csv"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path),  exist_ok=True)

    train_loader, val_loader, _, class_info = build_dataloaders(
        labels_csv=labels_csv,
        batch_size=cfg.get("batch_size", 32),
        num_workers=2,
        label_mode="combined",
        aug_config=cfg.get("augmentation", {}),
    )

    model = build_model(model_name, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    history = []

    # ── ResNet two-phase ──────────────────────────────────────────────────────
    if model_name == "resnet50":
        # Phase 1
        print("\nPhase 1: head only (backbone frozen)")
        opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        for epoch in range(1, 6):
            tl, ta = run_epoch(model, train_loader, criterion, opt, device, True)
            vl, va = run_epoch(model, val_loader,   criterion, opt, device, False)
            print(f"  [P1 E{epoch}] train {ta:.3f} | val {va:.3f}")
            history.append({"phase":1,"epoch":epoch,"train_loss":tl,"train_acc":ta,"val_loss":vl,"val_acc":va})

        # Phase 2
        print("\nPhase 2: top layers unfrozen")
        model.unfreeze_top_layers()
        opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
        es = EarlyStopping(patience=5, path=ckpt_path)
        for epoch in range(1, 16):
            tl, ta = run_epoch(model, train_loader, criterion, opt, device, True)
            vl, va = run_epoch(model, val_loader,   criterion, opt, device, False)
            print(f"  [P2 E{epoch}] train {ta:.3f} | val {va:.3f}")
            history.append({"phase":2,"epoch":epoch,"train_loss":tl,"train_acc":ta,"val_loss":vl,"val_acc":va})
            if es.step(vl, model, epoch):
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── Baseline and Improved ─────────────────────────────────────────────────
    else:
        lr = cfg.get("learning_rate", 0.001)
        wd = cfg.get("weight_decay", 0)
        opt_name = cfg.get("optimizer", "adam")
        epochs = cfg.get("epochs", 30)
        patience = cfg.get("patience", None)

        if opt_name == "adamw":
            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            opt = optim.Adam(model.parameters(), lr=lr)

        sched = None
        if cfg.get("lr_scheduler") == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        es = EarlyStopping(patience=patience or 999, path=ckpt_path)

        for epoch in range(1, epochs + 1):
            tl, ta = run_epoch(model, train_loader, criterion, opt, device, True)
            vl, va = run_epoch(model, val_loader,   criterion, opt, device, False)
            if sched: sched.step()
            lr_now = opt.param_groups[0]["lr"]
            print(f"  [E{epoch:03d}/{epochs}] train {ta:.3f} | val {va:.3f} | lr {lr_now:.6f}")
            history.append({"phase":1,"epoch":epoch,"train_loss":tl,"train_acc":ta,"val_loss":vl,"val_acc":va})
            if es.step(vl, model, epoch):
                print(f"  Early stopping at epoch {epoch} (best: {es.best_epoch})")
                break

        if not patience:
            torch.save(model.state_dict(), ckpt_path)

    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"\nCheckpoint → {ckpt_path}")
    print(f"Log        → {log_path}")
    return history