"""Extra training losses for the op experiments (LLMRouterBench_op only).

Three additions on top of the historical ``BCE + cost_rank`` objective:

* ``listwise_selection_loss``: multi-positive listwise CE over the correct
  models' probability mass, directly aligned with top-1 selection accuracy.
* ``decision_aligned_loss``: differentiable version of the deployed rule
  ``argmax(sigmoid(margin) - lambda * scaled_cost)`` with a per-batch sampled
  lambda, so one router serves several cost weights.
* task classification auxiliary loss is applied in the training loop directly
  (``router.task_classifier``), not here.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


NEG_INF = float("-inf")


def listwise_selection_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    supervision_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """-log(probability mass on correct models) per supervised view."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    margins = (logits[:, :, 1] - logits[:, :, 0]) / temperature
    log_probs = torch.log_softmax(margins, dim=1)
    correct = (labels > 0.5) & supervision_mask
    has_correct = correct.any(dim=1)
    if not bool(has_correct.any()):
        return logits.sum() * 0.0, 0
    masked = torch.where(correct, log_probs, torch.full_like(log_probs, NEG_INF))
    log_mass = torch.logsumexp(masked, dim=1)
    return -log_mass[has_correct].mean(), int(has_correct.sum().item())


def _row_scaled_costs(costs: torch.Tensor, rows: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """log1p costs, shifted to row minimum and scaled to [0, 1] per question."""
    usable = costs[rows].to(dtype=dtype)
    scaled = torch.log1p(usable.clamp_min(0.0))
    shifted = scaled - scaled.min(dim=1, keepdim=True).values
    span = shifted.max(dim=1, keepdim=True).values
    return shifted / (span + 1e-6)


def decision_aligned_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    costs: torch.Tensor,
    supervision_mask: torch.Tensor,
    routing_mask: torch.Tensor,
    cost_available_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    cost_weight: float = 0.3,
) -> tuple[torch.Tensor, int]:
    """Differentiable surrogate of the deployed cost-aware selection rule."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    usable_rows = (cost_available_mask | ~routing_mask).all(dim=1) \
        & routing_mask.any(dim=1) & (supervision_mask | ~routing_mask).all(dim=1)
    if not bool(usable_rows.any()):
        return logits.sum() * 0.0, 0
    rows = usable_rows.nonzero(as_tuple=False).squeeze(1)
    margins = (logits[:, :, 1] - logits[:, :, 0]) / temperature
    scaled_costs = torch.zeros_like(margins)
    scaled_costs[rows] = _row_scaled_costs(costs, rows, margins.dtype)
    scores = margins - cost_weight * scaled_costs
    log_probs = torch.log_softmax(scores, dim=1)
    correct = (labels > 0.5) & supervision_mask
    has_correct = correct.any(dim=1) & usable_rows
    if not bool(has_correct.any()):
        return logits.sum() * 0.0, 0
    masked = torch.where(correct, log_probs, torch.full_like(log_probs, NEG_INF))
    log_mass = torch.logsumexp(masked, dim=1)
    return -log_mass[has_correct].mean(), int(has_correct.sum().item())


class LambdaSampler:
    """Deterministic lambda sampler for the decision-aligned loss."""

    def __init__(self, values: tuple[float, ...] = (0.0, 0.1, 0.3, 1.0), seed: int = 42):
        if not values or any(value < 0 for value in values):
            raise ValueError("lambda values must be nonnegative")
        self.values = tuple(float(value) for value in values)
        self._rng = random.Random(seed)

    def next(self) -> float:
        return self._rng.choice(self.values)
