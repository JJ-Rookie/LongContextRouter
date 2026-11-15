from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

from baselines.SWERouter5.policy import estimate_remaining_costs, evaluate_policy


BENCHES = ("bbh", "memory_agent_bench", "mmau", "swe_bench_pro")
_TOLERANCE = 1e-9


def _cost(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _bench(row: Mapping) -> str:
    bench = row.get("bench")
    if bench not in BENCHES:
        raise ValueError(f"unknown four-benchmark identity: {bench!r}")
    return bench


def _validate_row(row: Mapping, explorer_index: int, width: int) -> str:
    bench = _bench(row)
    if type(row.get("prompt_id")) is not int or row["prompt_id"] < 0:
        raise ValueError("prompt_id must be a nonnegative integer")
    if not isinstance(row.get("instance_id"), str) or not row["instance_id"]:
        raise ValueError("instance_id must be nonempty text")
    labels, full_costs = row.get("labels"), row.get("full_costs")
    if not isinstance(labels, (list, tuple)) or len(labels) != width:
        raise ValueError("row has an incomplete five-model label pool")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("row has a nonbinary outcome")
    if not isinstance(full_costs, (list, tuple)) or len(full_costs) != width:
        raise ValueError("row has an incomplete five-model cost pool")
    costs = [_cost(value, f"full_cost[{index}]")
             for index, value in enumerate(full_costs)]
    prefix = _cost(row.get("prefix_cost"), "prefix cost")
    if prefix > costs[explorer_index] + _TOLERANCE:
        raise ValueError("Flash probe cost exceeds frozen Flash full cost")
    if bench != "swe_bench_pro" and not math.isclose(
        prefix, costs[explorer_index], rel_tol=0, abs_tol=_TOLERANCE
    ):
        raise ValueError("non-SWE Flash probe must cost its complete frozen response")
    if type(row.get("early_finish", False)) is not bool:
        raise ValueError("early_finish must be boolean")
    if bench != "swe_bench_pro" and row.get("early_finish", False):
        raise ValueError("non-SWE complete Flash response cannot have early_finish")
    return bench


def _pool_width(rows: Sequence[Mapping], explorer_index: int) -> int:
    if not rows:
        raise ValueError("rows are empty")
    costs = rows[0].get("full_costs")
    if not isinstance(costs, (list, tuple)) or len(costs) != 5:
        raise ValueError("five-model pool is required")
    if type(explorer_index) is not int or not 0 <= explorer_index < len(costs):
        raise ValueError("explorer index is outside the five-model pool")
    return len(costs)


def estimate_remaining_costs_by_bench(
    train_rows: Sequence[Mapping], *, explorer_index: int,
) -> dict[str, list[float]]:
    width = _pool_width(train_rows, explorer_index)
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    seen_ids: set[int] = set()
    for row in train_rows:
        bench = _validate_row(row, explorer_index, width)
        if row["prompt_id"] in seen_ids:
            raise ValueError("duplicate training prompt_id")
        seen_ids.add(row["prompt_id"])
        grouped[bench].append(row)
    if set(grouped) != set(BENCHES):
        raise ValueError("train rows must cover all four benchmarks")
    return {bench: estimate_remaining_costs(grouped[bench], explorer_index=explorer_index)
            for bench in BENCHES}


def evaluate_four_bench_policy(
    rows: Sequence[Mapping], predictions: Sequence[Sequence[float]],
    remaining_costs_by_bench: Mapping[str, Sequence[float]], cost_weight: float,
    *, explorer_index: int,
) -> dict:
    width = _pool_width(rows, explorer_index)
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must be nonempty and aligned")
    if not isinstance(remaining_costs_by_bench, Mapping):
        raise ValueError("remaining costs must be a benchmark mapping")
    if any(bench not in BENCHES for bench in remaining_costs_by_bench):
        raise ValueError("remaining costs contain an unknown benchmark")
    decisions = []
    seen_ids: set[int] = set()
    for row, predicted in zip(rows, predictions):
        bench = _validate_row(row, explorer_index, width)
        if row["prompt_id"] in seen_ids:
            raise ValueError("duplicate policy prompt_id")
        seen_ids.add(row["prompt_id"])
        costs = remaining_costs_by_bench.get(bench)
        if not isinstance(costs, (list, tuple)) or len(costs) != width:
            raise ValueError(f"{bench}: missing five-model remaining-cost priors")
        costs = [_cost(value, f"{bench} prior[{index}]")
                 for index, value in enumerate(costs)]
        replay = evaluate_policy([row], [predicted], costs, cost_weight,
                                 explorer_index=explorer_index)
        decisions.append({"bench": bench, **replay["decisions"][0]})
    per_bench = {}
    for bench in BENCHES:
        subset = [decision for decision in decisions if decision["bench"] == bench]
        if subset:
            successes = sum(item["success"] for item in subset)
            per_bench[bench] = {
                "count": len(subset), "successes": successes,
                "accuracy": successes / len(subset),
                "total_cost_usd": math.fsum(item["total_cost_usd"] for item in subset),
                "restart_count": sum(item["restart"] for item in subset),
            }
    successes = sum(item["success"] for item in decisions)
    return {
        "count": len(decisions), "successes": successes,
        "accuracy": successes / len(decisions),
        "total_cost_usd": math.fsum(item["total_cost_usd"] for item in decisions),
        "restart_count": sum(item["restart"] for item in decisions),
        "per_bench": per_bench, "decisions": decisions,
    }


def select_four_bench_validation_policy(
    validation_rows: Sequence[Mapping], predictions: Sequence[Sequence[float]],
    remaining_costs_by_bench: Mapping[str, Sequence[float]], *,
    weights: Sequence[float], budget_multiplier: float = 1.0,
    explorer_index: int,
) -> dict:
    _pool_width(validation_rows, explorer_index)
    if {_bench(row) for row in validation_rows} != set(BENCHES):
        raise ValueError("validation rows must cover all four benchmarks")
    multiplier = _cost(budget_multiplier, "budget_multiplier")
    if multiplier <= 0:
        raise ValueError("budget_multiplier must be positive")
    if not weights:
        raise ValueError("at least one candidate cost weight is required")
    baseline = math.fsum(_cost(row["full_costs"][explorer_index], "Flash full cost")
                         for row in validation_rows)
    if baseline <= 0:
        raise ValueError("frozen Flash validation baseline must be positive")
    budget = baseline * multiplier
    if not math.isfinite(budget):
        raise ValueError("validation budget must be finite")
    evaluated = [(weight, evaluate_four_bench_policy(
        validation_rows, predictions, remaining_costs_by_bench, weight,
        explorer_index=explorer_index)) for weight in weights]
    eligible = [(weight, report) for weight, report in evaluated
                if report["total_cost_usd"] <= budget + _TOLERANCE]
    if not eligible:
        raise ValueError("no four-benchmark validation policy satisfies the declared cost cap")
    weight, report = max(eligible, key=lambda item: (
        item[1]["successes"], -item[1]["total_cost_usd"], -item[0]))
    return {
        "cost_weight": weight, "baseline_cost_usd": baseline,
        "budget_usd": budget, "validation": report,
        "candidate_summary": [
            {"cost_weight": candidate, "successes": result["successes"],
             "total_cost_usd": result["total_cost_usd"],
             "within_budget": result["total_cost_usd"] <= budget + _TOLERANCE}
            for candidate, result in evaluated
        ],
    }
