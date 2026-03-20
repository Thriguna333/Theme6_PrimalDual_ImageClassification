"""
tests/test_optimizer.py  –  Unit tests for Theme 6 APAL Optimizer
Run: python tests/test_optimizer.py
(Requires: torch)
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from augmented_lagrangian import AugmentedLagrangianOptimizer
from model import ConstrainedResNet50


class TestAugmentedLagrangian(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model  = ConstrainedResNet50(num_classes=10).to(self.device)
        opt         = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.apal   = AugmentedLagrangianOptimizer(self.model, opt, rho_init=1.0)

    # ── dual variable initialization ──────────────────────────────────────────
    def test_mu_init_zeros(self):
        self.apal._ensure_mu(3, self.device)
        self.assertTrue(torch.allclose(self.apal._mu, torch.zeros(3)))

    # ── AL loss > base when constraints violated ───────────────────────────────
    def test_al_loss_penalty_on_violation(self):
        base = torch.tensor(1.0, requires_grad=True)
        g    = torch.tensor([0.5, 0.3, 0.2])
        self.apal._ensure_mu(3, self.device)
        al   = self.apal.augmented_lagrangian_loss(base, g)
        self.assertGreater(al.item(), base.item())

    # ── AL loss == base when constraints satisfied and mu=0 ───────────────────
    def test_al_loss_no_penalty_when_satisfied(self):
        base = torch.tensor(1.5)
        g    = torch.tensor([-0.5, -0.3, -0.1])
        self.apal._ensure_mu(3, self.device)
        al   = self.apal.augmented_lagrangian_loss(base, g)
        self.assertAlmostEqual(al.item(), base.item(), places=5)

    # ── dual update increases mu on violation ─────────────────────────────────
    def test_dual_update_increases_mu(self):
        self.apal._ensure_mu(3, self.device)
        g = torch.tensor([0.4, 0.2, 0.1])
        self.apal.dual_update(g)
        self.assertTrue((self.apal._mu[:3] > 0).all())

    # ── dual stays non-negative when satisfied ────────────────────────────────
    def test_dual_non_negative(self):
        self.apal._ensure_mu(3, self.device)
        g = torch.tensor([-0.5, -0.8, -0.2])
        self.apal.dual_update(g)
        self.assertTrue((self.apal._mu[:3] >= 0).all())

    # ── rho increases when primal dominates ───────────────────────────────────
    def test_rho_increase(self):
        self.apal.rho = 1.0
        self.apal._adapt_rho(100.0, 0.1)
        self.assertGreater(self.apal.rho, 1.0)

    # ── rho decreases when dual dominates ────────────────────────────────────
    def test_rho_decrease(self):
        self.apal.rho = 4.0
        self.apal._adapt_rho(0.01, 100.0)
        self.assertLess(self.apal.rho, 4.0)

    # ── rho bounded by rho_max ───────────────────────────────────────────────
    def test_rho_bounded_max(self):
        self.apal.rho = self.apal.rho_max
        self.apal._adapt_rho(1e8, 0.001)
        self.assertLessEqual(self.apal.rho, self.apal.rho_max)

    # ── rho bounded by rho_min ───────────────────────────────────────────────
    def test_rho_bounded_min(self):
        self.apal.rho = self.apal.rho_min
        self.apal._adapt_rho(0.001, 1e8)
        self.assertGreaterEqual(self.apal.rho, self.apal.rho_min)

    # ── model constraints return correct shape ───────────────────────────────
    def test_constraint_shape(self):
        _, g = self.model.get_constraints()
        self.assertEqual(g.shape, (3,))

    # ── forward pass shape ───────────────────────────────────────────────────
    def test_forward_shape(self):
        x   = torch.randn(4, 3, 32, 32)
        out = self.model(x)
        self.assertEqual(out.shape, (4, 10))

    # ── full step does not crash ──────────────────────────────────────────────
    def test_full_step(self):
        crit = nn.CrossEntropyLoss()
        x    = torch.randn(4, 3, 32, 32)
        y    = torch.randint(0, 10, (4,))
        self.apal.zero_grad()
        logits    = self.model(x)
        base_loss = crit(logits, y)
        _, g      = self.model.get_constraints()
        al_loss   = self.apal.augmented_lagrangian_loss(base_loss, g)
        al_loss.backward()
        self.apal.step()
        with torch.no_grad():
            _, g2 = self.model.get_constraints()
        self.apal.dual_update(g2)
        stats = self.apal.get_stats()
        self.assertIn("rho", stats)
        self.assertIn("constraint_violation", stats)

    # ── state_dict round-trip ─────────────────────────────────────────────────
    def test_state_dict(self):
        self.apal._ensure_mu(3, self.device)
        sd   = self.apal.state_dict()
        opt2 = torch.optim.SGD(self.model.parameters(), lr=0.01)
        a2   = AugmentedLagrangianOptimizer(self.model, opt2)
        a2.load_state_dict(sd)
        self.assertAlmostEqual(a2.rho, self.apal.rho)


if __name__ == "__main__":
    print("Running APAL unit tests…\n")
    unittest.main(verbosity=2)
