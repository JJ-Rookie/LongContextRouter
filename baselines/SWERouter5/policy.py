"""Five-candidate continuation/restart policy and offline replay metrics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def select_model(probabilities: Sequence[float], remaining_costs: Sequence[float],
                 cost_weight: float) -> int:
    """Choose by predicted success minus a cost penalty; lower ID breaks ties."""
    penalty = _nonnegative(cost_weight, "cost_weight")
    if not probabilities or len(probabilities) != len(remaining_costs):
        raise ValueError("probability and cost vectors must have equal nonzero width")
    scores = []
    for index, (probability, cost) in enumerate(zip(probabilities, remaining_costs)):
        p = _nonnegative(probability, f"probability[{index}]")
        if p > 1:
            raise ValueError("predicted success must not exceed one")
        scores.append(p - penalty * _nonnegative(cost, f"remaining_cost[{index}]"))
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def estimate_remaining_costs(rows: Sequence[Mapping], *, explorer_index: int) -> list[float]:
    """Fit model-cost expectations on training rows only."""
    if not rows:
        raise ValueError("training rows are empty")
    width = len(rows[0]["full_costs"])
    if not 0 <= explorer_index < width:
        raise ValueError("explorer index is outside the model pool")
    totals = [0.0] * width
    prefix_total = 0.0
    for row in rows:
        costs = row["full_costs"]
        if len(costs) != width:
            raise ValueError("training row has an incomplete model pool")
        if row.get("prefix_cost") is None:
            raise ValueError("training row is missing prefix cost")
        prefix_total += _nonnegative(row["prefix_cost"], "prefix cost")
        for index, cost in enumerate(costs):
            totals[index] += _nonnegative(cost, f"full_cost[{index}]")
    average = [total / len(rows) for total in totals]
    average[explorer_index] -= prefix_total / len(rows)
    if average[explorer_index] < -1e-9:
        raise ValueError("mean explorer prefix cost exceeds mean full cost")
    average[explorer_index] = max(0.0, average[explorer_index])
    return average


def evaluate_policy(
    rows: Sequence[Mapping], predictions: Sequence[Sequence[float]],
    remaining_costs: Sequence[float], cost_weight: float, *, explorer_index: int,
) -> dict:
    """Replay one fixed policy using observed branch outcomes and strict costs."""
    if len(rows) != len(predictions) or not rows:
        raise ValueError("rows and predictions must be nonempty and aligned")
    if not 0 <= explorer_index < len(remaining_costs):
        raise ValueError("explorer index is outside the model pool")
    decisions = []
    for row, predicted in zip(rows, predictions):
        labels, full_costs = row["labels"], row["full_costs"]
        if len(labels) != len(remaining_costs) or len(full_costs) != len(remaining_costs):
            raise ValueError("row has an incomplete model pool")
        if any(label not in (0, 1) for label in labels):
            raise ValueError("row has a nonbinary outcome")
        if row.get("prefix_cost") is None:
            raise ValueError(f"prefix cost is unavailable for {row['prompt_id']}")
        prefix = _nonnegative(row["prefix_cost"], "prefix cost")
        costs = [_nonnegative(value, f"full_cost[{index}]")
                 for index, value in enumerate(full_costs)]
        if prefix > costs[explorer_index] + 1e-9:
            raise ValueError("prefix cost exceeds explorer full-run cost")
        early = bool(row.get("early_finish", False))
        selected = explorer_index if early else select_model(predicted, remaining_costs, cost_weight)
        restart = selected != explorer_index
        charged = costs[selected] + (prefix if restart else 0.0)
        decisions.append({
            "prompt_id": row["prompt_id"], "instance_id": row["instance_id"],
            "selected_model_id": selected, "success": int(labels[selected]),
            "prefix_cost_usd": prefix, "selected_full_cost_usd": costs[selected],
            "total_cost_usd": charged, "restart": restart, "early_finish": early,
        })
    successes = sum(item["success"] for item in decisions)
    return {"count": len(decisions), "successes": successes,
            "accuracy": successes / len(decisions),
            "total_cost_usd": sum(item["total_cost_usd"] for item in decisions),
            "restart_count": sum(item["restart"] for item in decisions),
            "decisions": decisions}


def select_validation_policy(
    rows: Sequence[Mapping], predictions: Sequence[Sequence[float]],
    remaining_costs: Sequence[float], *, weights: Sequence[float],
    max_cost_usd: float, explorer_index: int,
) -> dict:
    """Lock the most accurate validation policy within a predeclared cost cap."""
    cap = _nonnegative(max_cost_usd, "max_cost_usd")
    if not weights:
        raise ValueError("at least one candidate cost weight is required")
    evaluated = [(float(weight), evaluate_policy(
        rows, predictions, remaining_costs, weight, explorer_index=explorer_index))
        for weight in weights]
    eligible = [(weight, report) for weight, report in evaluated
                if report["total_cost_usd"] <= cap + 1e-9]
    if not eligible:
        raise ValueError("no validation policy satisfies the declared cost cap")
    weight, report = max(eligible, key=lambda item: (
        item[1]["successes"], -item[1]["total_cost_usd"], -item[0]))
    return {"cost_weight": weight, "validation": report,
            "candidate_summary": [{"cost_weight": candidate,
                                   "successes": result["successes"],
                                   "total_cost_usd": result["total_cost_usd"],
                                   "within_budget": result["total_cost_usd"] <= cap + 1e-9}
                                  for candidate, result in evaluated]}
