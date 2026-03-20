"""
model.py
========
Theme 6 — Constrained ResNet-50 for CIFAR-10

Architecture modifications from standard ResNet-50 for 32×32 inputs:
  • Stem: 3×3 conv (stride=1, no max-pool) instead of 7×7 conv + max-pool
    This preserves spatial resolution for small images (standard practice,
    see He et al. 2016 CIFAR experiments).
  • All other bottleneck blocks unchanged.

Constraints enforced via Augmented Lagrangian (Theme 6):
  g1 : L1 sparsity on conv layer weights  → mean|w| ≤ κ
  g2 : Frobenius-norm bound on FC weights → ‖W_fc‖²_F / d ≤ γ
  g3 : Spectral-norm proxy on final layer → ‖W_fc‖_∞ ≤ δ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Standard ResNet-50 building blocks
# ─────────────────────────────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1  = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(planes)
        self.conv2  = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes)
        self.conv3  = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3    = nn.BatchNorm2d(planes * 4)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)


# ─────────────────────────────────────────────────────────────────────────────
# Constrained ResNet-50 (CIFAR-10 variant)
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainedResNet50(nn.Module):
    """
    ResNet-50 adapted for CIFAR-10 with Augmented Lagrangian constraints.

    Parameters
    ----------
    num_classes     : int   – number of output classes (10 for CIFAR-10)
    sparsity_budget : float – κ in g1: mean|w_conv| ≤ κ
    norm_budget     : float – γ in g2: ‖W_fc‖²_F / d ≤ γ
    spectral_budget : float – δ in g3: max row-norm of W_fc ≤ δ
    dropout_rate    : float – dropout before classifier
    """

    def __init__(
        self,
        num_classes:     int   = 10,
        sparsity_budget: float = 0.3,
        norm_budget:     float = 5.0,
        spectral_budget: float = 2.0,
        dropout_rate:    float = 0.3,
    ):
        super().__init__()
        self.sparsity_budget = sparsity_budget
        self.norm_budget     = norm_budget
        self.spectral_budget = spectral_budget
        self.in_planes       = 64

        # ── CIFAR-10 stem: 3×3, stride=1, no max-pool ──────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── ResNet-50 stages ─────────────────────────────────────────────────
        self.layer1 = self._make_layer(64,  3, stride=1)   # 32×32
        self.layer2 = self._make_layer(128, 4, stride=2)   # 16×16
        self.layer3 = self._make_layer(256, 6, stride=2)   #  8×8
        self.layer4 = self._make_layer(512, 3, stride=2)   #  4×4

        # ── Classifier ────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout_rate)
        self.fc      = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers  = [Bottleneck(self.in_planes, planes, stride)]
        self.in_planes = planes * Bottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    # ------------------------------------------------------------------ #
    #  Constraint functions g_i(θ)  ≤ 0                                   #
    # ------------------------------------------------------------------ #
    def get_constraints(self) -> Tuple[None, torch.Tensor]:
        """
        Compute inequality constraint residuals for APAL.

        Returns (None, ineq_tensor) where ineq_tensor has shape (3,):
            g1 = mean|w_conv| − κ       (L1 sparsity)
            g2 = ‖W_fc‖²_F/d  − γ      (Frobenius norm)
            g3 = max_row_norm(W_fc) − δ (spectral proxy)
        """
        constraints = []

        # g1 – L1 sparsity across all conv weights in stem + layer1–4
        conv_modules = [
            self.stem[0],
            *[m for layer in [self.layer1, self.layer2, self.layer3, self.layer4]
              for m in layer.modules() if isinstance(m, nn.Conv2d)],
        ]
        all_conv_w = torch.cat([m.weight.flatten() for m in conv_modules])
        g1 = torch.mean(torch.abs(all_conv_w)) - self.sparsity_budget
        constraints.append(g1.unsqueeze(0))

        # g2 – Frobenius norm bound on FC layer
        fc_w = self.fc.weight
        g2   = torch.norm(fc_w, p="fro") ** 2 / fc_w.numel() - self.norm_budget
        constraints.append(g2.unsqueeze(0))

        # g3 – Spectral proxy: max row norm of FC weight
        row_norms = torch.norm(fc_w, dim=1)          # shape (num_classes,)
        g3        = torch.max(row_norms) - self.spectral_budget
        constraints.append(g3.unsqueeze(0))

        return None, torch.cat(constraints)          # shape (3,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
