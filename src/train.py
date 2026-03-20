"""
train.py
========
Theme 6 — Full Training Pipeline: APAL + ResNet-50 on CIFAR-10

Techniques used to reach ≥ 95% test accuracy
---------------------------------------------
1. ResNet-50 with CIFAR-10 stem (3×3 conv, no max-pool)
2. Strong augmentation: RandomCrop + HorizFlip + AutoAugment(CIFAR10)
   + CutMix (α=1.0) applied with probability 0.5
3. Label Smoothing cross-entropy (ε=0.1)
4. SGD + Nesterov momentum (lr=0.1) with Warm-up (5 epochs) +
   Cosine Annealing decay to 1e-4 over 100 epochs
5. Weight decay 5e-4 on conv/bn, NO weight decay on bias/BN params
6. APAL dual update: once per epoch, adaptive ρ (τ=2, ξ=10)

Usage
-----
  # Train all methods:
  python src/train.py --method augmented_lagrangian --epochs 100
  python src/train.py --method adam_baseline        --epochs 100
  python src/train.py --method penalty_method       --epochs 100

  # Then visualise:
  python src/visualize.py --results_dir ./results
"""

import os, json, time, math, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Tuple, Optional

from model import ConstrainedResNet50
from augmented_lagrangian import AugmentedLagrangianOptimizer


# ─────────────────────────────────────────────────────────────────────────────
# CutMix utility
# ─────────────────────────────────────────────────────────────────────────────

def rand_bbox(size: Tuple, lam: float):
    W, H   = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w   = int(W * cut_rat)
    cut_h   = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    lam     = np.random.beta(alpha, alpha)
    bs      = x.size(0)
    idx     = torch.randperm(bs, device=x.device)
    x2, y2  = x[idx], y[idx]
    x1, y1, x2c, y2c = rand_bbox(x.size(), lam)
    x_mix   = x.clone()
    x_mix[:, :, x1:x2c, y1:y2c] = x2[:, :, x1:x2c, y1:y2c]
    lam_adj = 1 - (x2c - x1) * (y2c - y1) / (x.size(-1) * x.size(-2))
    return x_mix, y, y2, lam_adj


# ─────────────────────────────────────────────────────────────────────────────
# Label-Smoothing Cross-Entropy
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothingCE(nn.Module):
    def __init__(self, classes: int = 10, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.cls       = classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        log_probs  = torch.log_softmax(pred, dim=1)
        nll        = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        return (confidence * nll + self.smoothing * smooth_loss).mean()


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(
    data_dir:   str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    try:
        aa = transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            aa,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    except AttributeError:
        # Older torchvision fallback
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
        ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Warmup + Cosine Annealing scheduler
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float, min_lr: float = 1e-4):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self._epoch        = 0

    def step(self):
        self._epoch += 1
        e = self._epoch
        if e <= self.warmup_epochs:
            lr = self.base_lr * e / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr       = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ─────────────────────────────────────────────────────────────────────────────
# Separate parameter groups (no weight-decay on BN/bias)
# ─────────────────────────────────────────────────────────────────────────────

def make_param_groups(model: nn.Module, wd: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name or "bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch – APAL
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch_apal(
    model:      ConstrainedResNet50,
    apal:       AugmentedLagrangianOptimizer,
    criterion:  nn.Module,
    loader,
    device:     torch.device,
    cutmix_prob: float = 0.5,
) -> Dict:
    model.train()
    total_loss, base_total = 0.0, 0.0
    correct, total = 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # CutMix augmentation
        use_cutmix = np.random.rand() < cutmix_prob
        if use_cutmix:
            inputs, y_a, y_b, lam = cutmix_data(inputs, labels)

        apal.zero_grad()
        logits    = model(inputs)

        if use_cutmix:
            base_loss = cutmix_criterion(criterion, logits, y_a, y_b, lam)
        else:
            base_loss = criterion(logits, labels)

        _, ineq_c = model.get_constraints()
        al_loss   = apal.augmented_lagrangian_loss(base_loss, ineq_constraints=ineq_c)

        al_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        apal.step()

        total_loss  += al_loss.item()  * inputs.size(0)
        base_total  += base_loss.item() * inputs.size(0)
        _, pred      = logits.max(1)
        correct     += pred.eq(labels).sum().item()
        total       += inputs.size(0)

    # Dual update (once per epoch on full-batch constraint)
    with torch.no_grad():
        _, ineq_c2 = model.get_constraints()
    apal.dual_update(ineq_c2)

    n = len(loader.dataset)
    return {
        "loss":      total_loss / n,
        "base_loss": base_total / n,
        "accuracy":  100.0 * correct / total,
        **apal.get_stats(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch – Baseline
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch_baseline(
    model:        nn.Module,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    loader,
    device:       torch.device,
    penalty_lam:  float = 0.0,
    cutmix_prob:  float = 0.5,
) -> Dict:
    model.train()
    total_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        use_cutmix = np.random.rand() < cutmix_prob
        if use_cutmix:
            inputs, y_a, y_b, lam = cutmix_data(inputs, labels)

        optimizer.zero_grad()
        logits = model(inputs)

        if use_cutmix:
            loss = cutmix_criterion(criterion, logits, y_a, y_b, lam)
        else:
            loss = criterion(logits, labels)

        if penalty_lam > 0.0:
            l2 = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
            loss = loss + penalty_lam * l2

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, pred     = logits.max(1)
        correct    += pred.eq(labels).sum().item()
        total      += inputs.size(0)

    return {
        "loss":     total_loss / len(loader.dataset),
        "accuracy": 100.0 * correct / total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader, device: torch.device) -> Dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits  = model(inputs)
            loss    = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            _, pred  = logits.max(1)
            correct += pred.eq(labels).sum().item()
            total   += inputs.size(0)

    n = len(loader.dataset)
    return {"test_loss": total_loss / n, "test_accuracy": 100.0 * correct / total}


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    method:        str   = "augmented_lagrangian",
    epochs:        int   = 100,
    batch_size:    int   = 128,
    base_lr:       float = 0.1,
    weight_decay:  float = 5e-4,
    rho_init:      float = 1.0,
    penalty_lam:   float = 1e-4,
    warmup_epochs: int   = 5,
    cutmix_prob:   float = 0.5,
    data_dir:      str   = "./data",
    save_dir:      str   = "./results",
    num_workers:   int   = 4,
    resume:        Optional[str] = None,
) -> Dict:

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  METHOD : {method.upper()}")
    print(f"  DEVICE : {device}  |  EPOCHS : {epochs}  |  LR : {base_lr}")
    print(f"{'='*65}\n")

    train_loader, test_loader = get_loaders(data_dir, batch_size, num_workers)

    model     = ConstrainedResNet50(num_classes=10, sparsity_budget=0.3,
                                     norm_budget=5.0, spectral_budget=2.0).to(device)
    criterion = LabelSmoothingCE(classes=10, smoothing=0.1)

    print(f"  ResNet-50 CIFAR-10 | Parameters: {model.count_parameters():,}\n")

    history = {
        "method": method, "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [], "lr": [], "epoch_time": [],
    }

    # ── APAL ────────────────────────────────────────────────────────────────
    if method == "augmented_lagrangian":
        pg        = make_param_groups(model, weight_decay)
        inner_opt = optim.SGD(pg, lr=base_lr, momentum=0.9, nesterov=True)
        apal      = AugmentedLagrangianOptimizer(model, inner_opt, rho_init=rho_init)
        scheduler = WarmupCosineScheduler(inner_opt, warmup_epochs, epochs, base_lr)

        for key in ["rho", "primal_residual", "dual_residual", "constraint_violation"]:
            history[key] = []

        start_epoch = 1
        if resume and os.path.exists(resume):
            ckpt = torch.load(resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            inner_opt.load_state_dict(ckpt["optimizer"])
            apal.load_state_dict(ckpt["apal"])
            history     = ckpt["history"]
            start_epoch = ckpt["epoch"] + 1
            print(f"  Resumed from epoch {start_epoch-1}")

        for epoch in range(start_epoch, epochs + 1):
            t0   = time.time()
            lr   = scheduler.step()
            stat = train_epoch_apal(model, apal, criterion, train_loader,
                                     device, cutmix_prob)
            test = evaluate(model, test_loader, device)
            et   = time.time() - t0

            history["train_loss"].append(stat["loss"])
            history["train_acc"].append(stat["accuracy"])
            history["test_loss"].append(test["test_loss"])
            history["test_acc"].append(test["test_accuracy"])
            history["lr"].append(lr)
            history["epoch_time"].append(et)
            history["rho"].append(stat["rho"])
            history["primal_residual"].append(stat["primal_residual"])
            history["dual_residual"].append(stat["dual_residual"])
            history["constraint_violation"].append(stat["constraint_violation"])

            print(
                f"  Ep [{epoch:3d}/{epochs}] "
                f"LR:{lr:.5f} | Loss:{stat['base_loss']:.4f} | "
                f"TrainAcc:{stat['accuracy']:.2f}% | TestAcc:{test['test_accuracy']:.2f}% | "
                f"ρ:{stat['rho']:.3f} | CV:{stat['constraint_violation']:.4f} | "
                f"t:{et:.0f}s"
            )

            # Checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": inner_opt.state_dict(), "apal": apal.state_dict(),
                    "history": history,
                }, os.path.join(save_dir, f"{method}_ckpt_ep{epoch}.pth"))

    # ── BASELINES ────────────────────────────────────────────────────────────
    else:
        pg        = make_param_groups(model, weight_decay)
        optimizer = optim.SGD(pg, lr=base_lr, momentum=0.9, nesterov=True)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, base_lr)
        pl        = penalty_lam if method == "penalty_method" else 0.0

        for epoch in range(1, epochs + 1):
            t0   = time.time()
            lr   = scheduler.step()
            stat = train_epoch_baseline(model, optimizer, criterion,
                                         train_loader, device, pl, cutmix_prob)
            test = evaluate(model, test_loader, device)
            et   = time.time() - t0

            history["train_loss"].append(stat["loss"])
            history["train_acc"].append(stat["accuracy"])
            history["test_loss"].append(test["test_loss"])
            history["test_acc"].append(test["test_accuracy"])
            history["lr"].append(lr)
            history["epoch_time"].append(et)

            print(
                f"  Ep [{epoch:3d}/{epochs}] "
                f"LR:{lr:.5f} | Loss:{stat['loss']:.4f} | "
                f"TrainAcc:{stat['accuracy']:.2f}% | TestAcc:{test['test_accuracy']:.2f}% | "
                f"t:{et:.0f}s"
            )

    # Save final results
    out_path = os.path.join(save_dir, f"{method}_history.json")
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{method}_model.pth"))

    print(f"\n  ✅  Final Test Accuracy : {history['test_acc'][-1]:.2f}%")
    print(f"  Results → {save_dir}\n")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Theme 6 — APAL + ResNet-50 on CIFAR-10"
    )
    parser.add_argument("--method",       default="augmented_lagrangian",
                        choices=["augmented_lagrangian","adam_baseline","penalty_method"])
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--rho_init",     type=float, default=1.0)
    parser.add_argument("--penalty_lam",  type=float, default=1e-4)
    parser.add_argument("--warmup",       type=int,   default=5)
    parser.add_argument("--cutmix_prob",  type=float, default=0.5)
    parser.add_argument("--data_dir",     default="./data")
    parser.add_argument("--save_dir",     default="./results")
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--resume",       default=None)
    args = parser.parse_args()

    run_experiment(
        method        = args.method,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        base_lr       = args.lr,
        weight_decay  = args.weight_decay,
        rho_init      = args.rho_init,
        penalty_lam   = args.penalty_lam,
        warmup_epochs = args.warmup,
        cutmix_prob   = args.cutmix_prob,
        data_dir      = args.data_dir,
        save_dir      = args.save_dir,
        num_workers   = args.num_workers,
        resume        = args.resume,
    )
