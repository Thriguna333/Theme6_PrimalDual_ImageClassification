"""
augmented_lagrangian.py
=======================
Theme 6 — Primal-Dual Updating Schemes via Augmented Lagrangians
Adaptive Primal-Dual Augmented Lagrangian (APAL) Optimizer.

Mathematical formulation
------------------------
Problem:
    min_θ  L(θ)    s.t.  g_i(θ) ≤ 0,  i = 1,…,m

Augmented Lagrangian (exact penalty / PHR form):
    L_ρ(θ,μ) = L(θ) + (1/2ρ) Σ_i { max(0, μ_i + ρ·g_i(θ))² − μ_i² }

Dual (multiplier) update:
    μ_i^(k+1) = max(0, μ_i^(k) + ρ·g_i(θ^(k)))

Adaptive penalty parameter (Boyd et al., 2011):
    ρ ← ρ·τ   if  ‖r_primal‖ > ξ·‖r_dual‖
    ρ ← ρ/τ   if  ‖r_dual‖   > ξ·‖r_primal‖
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple


class AugmentedLagrangianOptimizer:
    """
    Adaptive Primal-Dual Augmented Lagrangian optimizer wrapper.

    Wraps any standard PyTorch optimizer as the inner primal solver
    and adds dual-ascent updates for inequality constraints.

    Parameters
    ----------
    model          : nn.Module  – network being optimized
    base_optimizer : Optimizer  – inner primal optimizer (e.g. SGD/Adam)
    rho_init       : float      – initial penalty parameter ρ₀
    tau            : float      – penalty scale factor τ > 1
    xi             : float      – residual imbalance threshold ξ
    rho_max        : float      – upper bound for ρ
    rho_min        : float      – lower bound for ρ
    dual_lr        : float      – optional learning-rate scaling for dual update
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        rho_init: float = 1.0,
        tau: float = 2.0,
        xi: float = 10.0,
        rho_max: float = 1e4,
        rho_min: float = 1e-4,
        dual_lr: float = 1.0,
    ):
        self.model          = model
        self.optimizer      = base_optimizer
        self.rho            = rho_init
        self.tau            = tau
        self.xi             = xi
        self.rho_max        = rho_max
        self.rho_min        = rho_min
        self.dual_lr        = dual_lr

        # Dual variables (Lagrange multipliers) – lazy init
        self._mu: Optional[torch.Tensor] = None
        self._n_ineq: int = 0

        # Logging
        self.rho_history:           List[float] = [rho_init]
        self.primal_res_history:    List[float] = []
        self.dual_res_history:      List[float] = []
        self.constraint_viol_history: List[float] = []

    # ------------------------------------------------------------------ #
    #  Dual variable management                                            #
    # ------------------------------------------------------------------ #
    def _ensure_mu(self, n: int, device: torch.device):
        if self._mu is None or self._n_ineq != n:
            self._mu    = torch.zeros(n, device=device, dtype=torch.float32)
            self._n_ineq = n

    # ------------------------------------------------------------------ #
    #  Augmented Lagrangian loss computation (forward)                     #
    # ------------------------------------------------------------------ #
    def augmented_lagrangian_loss(
        self,
        base_loss: torch.Tensor,
        ineq_constraints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute L_ρ(θ,μ) = L(θ) + AL penalty terms.

        Parameters
        ----------
        base_loss        : scalar cross-entropy loss (already computed)
        ineq_constraints : shape (m,) tensor of g_i(θ) values

        Returns
        -------
        Augmented Lagrangian total loss (scalar, differentiable w.r.t. θ)
        """
        if ineq_constraints is None:
            return base_loss

        g      = ineq_constraints
        device = g.device
        n      = g.numel()
        self._ensure_mu(n, device)

        # PHR formula: (1/2ρ) Σ [max(0, μ+ρg)² − μ²]
        mu        = self._mu[:n]
        aug_term  = torch.clamp(mu + self.rho * g.flatten(), min=0.0)
        al_penalty = (1.0 / (2.0 * self.rho)) * (
            torch.sum(aug_term ** 2) - torch.sum(mu ** 2)
        )
        return base_loss + al_penalty

    # ------------------------------------------------------------------ #
    #  Dual (multiplier) update – called once per epoch                   #
    # ------------------------------------------------------------------ #
    def dual_update(self, ineq_constraints: torch.Tensor):
        """
        Update Lagrange multipliers μ via projected dual ascent.

        μ_i^(k+1) = max(0, μ_i^(k) + ρ · g_i(θ^(k)))
        """
        with torch.no_grad():
            g  = ineq_constraints.detach().flatten()
            n  = g.numel()
            self._ensure_mu(n, g.device)
            mu = self._mu[:n]

            old_mu    = mu.clone()
            new_mu    = torch.clamp(old_mu + self.dual_lr * self.rho * g, min=0.0)
            self._mu[:n] = new_mu

            # Residuals
            primal_res = torch.norm(torch.clamp(g, min=0.0)).item()
            dual_res   = torch.norm(new_mu - old_mu).item()
            viol       = primal_res

            self.primal_res_history.append(primal_res)
            self.dual_res_history.append(dual_res)
            self.constraint_viol_history.append(viol)

            self._adapt_rho(primal_res, dual_res)

    # ------------------------------------------------------------------ #
    #  Adaptive penalty update                                             #
    # ------------------------------------------------------------------ #
    def _adapt_rho(self, primal_res: float, dual_res: float):
        eps = 1e-10
        ratio = primal_res / (dual_res + eps)
        if ratio > self.xi:
            self.rho = min(self.rho * self.tau, self.rho_max)
        elif ratio < 1.0 / self.xi:
            self.rho = max(self.rho / self.tau, self.rho_min)
        self.rho_history.append(self.rho)

    # ------------------------------------------------------------------ #
    #  Primal step delegation                                              #
    # ------------------------------------------------------------------ #
    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #
    def get_stats(self) -> Dict:
        return {
            "rho":                self.rho,
            "primal_residual":    self.primal_res_history[-1]  if self.primal_res_history  else 0.0,
            "dual_residual":      self.dual_res_history[-1]    if self.dual_res_history    else 0.0,
            "constraint_violation": self.constraint_viol_history[-1] if self.constraint_viol_history else 0.0,
        }

    def state_dict(self) -> Dict:
        return {
            "rho":   self.rho,
            "mu":    self._mu.cpu() if self._mu is not None else None,
            "n_ineq": self._n_ineq,
            "rho_history": self.rho_history,
        }

    def load_state_dict(self, sd: Dict):
        self.rho      = sd["rho"]
        self._n_ineq  = sd.get("n_ineq", 0)
        self.rho_history = sd.get("rho_history", [self.rho])
        if sd["mu"] is not None:
            self._mu = sd["mu"]
