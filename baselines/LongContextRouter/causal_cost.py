from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import median


COST_SCALES = ("minmax", "train_median", "train_log1p_median")


def _number(value, name, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError(f"{name} must be a finite{' nonnegative' if nonnegative else ''} number")
    return float(value)


def _models(models):
    if (not isinstance(models, (list, tuple)) or not models
            or any(not isinstance(model, str) or not model.strip() for model in models)
            or len(set(models)) != len(models)):
        raise ValueError("models must be a nonempty ordered sequence of unique names")
    return tuple(models)


def _vector(values, count, name, nonnegative=False):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} must contain exactly {count} values")
    return tuple(_number(value, name, nonnegative) for value in values)


def _benchmark(category):
    if not isinstance(category, str) or not category.strip() or not category.split("/", 1)[0].strip():
        raise ValueError("category must name a benchmark")
    return category.split("/", 1)[0]


def _provenance(provenance):
    if not isinstance(provenance, Mapping) or provenance.get("fit_split") != "train":
        raise ValueError("Profile provenance must declare fit_split='train'")
    try:
        return json.loads(json.dumps(dict(provenance), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Profile provenance must be finite JSON data") from error


def fit_profile(records, models, provenance):
    names, origin = _models(models), _provenance(provenance)
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("Training records must be a nonempty list or tuple")
    seen, grouped = set(), {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Each training record must be a mapping")
        identifier = record.get("prompt_id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("Each original training group requires a unique nonempty prompt_id")
        seen.add(identifier)
        bench = _benchmark(record.get("category"))
        costs = _vector(record.get("costs"), len(names), "costs", nonnegative=True)
        grouped.setdefault(bench, []).append(costs)
    means = {bench: {"mean_costs": [math.fsum(row[index] / len(rows) for row in rows)
                                   for index in range(len(names))],
                     "counts": [len(rows)] * len(names)}
             for bench, rows in sorted(grouped.items())}
    fitted = {"schema_version": 1, "mode": "train_task_mean", "cost_unit": "whole_problem",
              "models": list(names), "by_benchmark": means, "provenance": origin}
    return validate_profile(fitted, names)


def validate_profile(profile, models):
    names = _models(models)
    if not isinstance(profile, Mapping):
        raise ValueError("Cost profile must be a mapping")
    if type(profile.get("schema_version")) is not int or profile["schema_version"] != 1:
        raise ValueError("Unsupported cost-profile schema_version")
    if profile.get("mode") != "train_task_mean" or profile.get("cost_unit") != "whole_problem":
        raise ValueError("Cost profile must use train_task_mean and whole_problem costs")
    if _models(profile.get("models")) != names:
        raise ValueError("Cost profile model names and order must match the router")
    origin, benchmarks = _provenance(profile.get("provenance")), profile.get("by_benchmark")
    if not isinstance(benchmarks, Mapping) or not benchmarks:
        raise ValueError("Cost profile must contain benchmark statistics")
    validated = {}
    for bench, stats in benchmarks.items():
        if _benchmark(bench) != bench or not isinstance(stats, Mapping):
            raise ValueError("Profile benchmark keys must be root categories with statistics")
        means = _vector(stats.get("mean_costs"), len(names), "mean_costs", nonnegative=True)
        counts = stats.get("counts")
        if (not isinstance(counts, (list, tuple)) or len(counts) != len(names)
                or any(type(count) is not int or count < 1 for count in counts)
                or len(set(counts)) != 1):
            raise ValueError("Each profile model must have the same positive original-group count")
        validated[bench] = {"mean_costs": list(means), "counts": list(counts)}
    return {"schema_version": 1, "mode": "train_task_mean", "cost_unit": "whole_problem",
            "models": list(names), "by_benchmark": validated, "provenance": origin}


def _unique_json_keys(pairs):
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate JSON key in cost profile")
    return dict(pairs)


def load_profile(path, models):
    with Path(path).open(encoding="utf-8") as handle:
        profile = json.load(handle, object_pairs_hook=_unique_json_keys)
    return validate_profile(profile, models)


def estimate_costs(profile, category):
    if not isinstance(profile, Mapping):
        raise ValueError("A train-only cost profile is required")
    checked = validate_profile(profile, profile.get("models"))
    benchmark = _benchmark(category)
    if benchmark not in checked["by_benchmark"]:
        raise ValueError(f"Cost profile has no benchmark {benchmark!r}")
    return tuple(checked["by_benchmark"][benchmark]["mean_costs"])


def selection_costs(*, category, models, actual_costs=None, cost_mode="actual", cost_profile=None):
    names = _models(models)
    if cost_mode == "train_task_mean":
        checked = validate_profile(cost_profile, names)
        return estimate_costs(checked, category)
    if cost_mode == "actual":
        return _vector(actual_costs, len(names), "actual_costs", nonnegative=True)
    raise ValueError("cost_mode must be 'actual' or 'train_task_mean'")


def _spec(task_policy, benchmark, count):
    if not isinstance(task_policy, Mapping) or benchmark not in task_policy:
        raise ValueError(f"Enabled task policy has no specification for {benchmark!r}")
    spec = task_policy[benchmark]
    if not isinstance(spec, Mapping) or "weight" not in spec:
        raise ValueError("Task specification must contain weight")
    if set(spec) - {"weight", "anchor", "threshold", "anchor_only"}:
        raise ValueError("Unknown task specification fields")
    weight = _number(spec["weight"], "weight", nonnegative=True)
    anchor, threshold, only = spec.get("anchor"), spec.get("threshold"), spec.get("anchor_only", False)
    if anchor is not None and (type(anchor) is not int or anchor not in range(count)):
        raise ValueError("anchor must be None or a valid candidate model index")
    if type(only) is not bool:
        raise ValueError("anchor_only must be boolean")
    if (only or threshold is not None) and anchor is None:
        raise ValueError("Guard threshold and anchor_only require an anchor")
    if threshold is not None:
        threshold = _number(threshold, "threshold", nonnegative=True)
    return weight, anchor, threshold, only


def _sigmoid(score):
    if score >= 0:
        return 1. / (1. + math.exp(-score))
    exponential = math.exp(score)
    return exponential / (1. + exponential)


def _scale_name(cost_scale):
    if not isinstance(cost_scale, str) or cost_scale not in COST_SCALES:
        raise ValueError(f"cost_scale must be one of {COST_SCALES}")
    return cost_scale


def _scale_scope(cost_scale, cost_mode, benches):
    scale = _scale_name(cost_scale)
    if scale != "minmax" and cost_mode != "train_task_mean":
        raise ValueError("Training cost scales require cost_mode='train_task_mean'")
    if benches is None:
        return None
    if scale == "minmax":
        raise ValueError("cost_scale_benches requires a non-minmax cost scale")
    if (not isinstance(benches, (list, tuple)) or not benches
            or any(not isinstance(bench, str) or not bench or bench.strip() != bench
                   or "/" in bench for bench in benches)
            or len(set(benches)) != len(benches)):
        raise ValueError("cost_scale_benches must contain unique nonempty root benchmark names")
    return tuple(benches)


def _log1p_ratio(cost, scale):
    ratio = cost / scale
    if math.isfinite(ratio):
        return math.log1p(ratio)
    log_ratio = math.log(cost) - math.log(scale)
    return log_ratio + math.log1p(math.exp(-log_ratio))


def scaled_costs(costs, cost_scale="minmax"):
    scale_name = _scale_name(cost_scale)
    if not isinstance(costs, (list, tuple)) or not costs:
        raise ValueError("costs must be a nonempty list or tuple")
    values = _vector(costs, len(costs), "costs", nonnegative=True)
    minimum, spread = min(values), max(values) - min(values)
    if spread == 0:
        return (0.,) * len(values)
    if scale_name == "minmax":
        return tuple((cost - minimum) / spread for cost in values)
    denominator = median(tuple(value for value in values if value > 0))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Training cost median is not finite and positive")
    if scale_name == "train_median":
        penalties = tuple((cost - minimum) / denominator for cost in values)
    else:
        baseline = _log1p_ratio(minimum, denominator)
        penalties = tuple(_log1p_ratio(cost, denominator) - baseline for cost in values)
    if any(not math.isfinite(value) or value < 0 for value in penalties):
        raise ValueError("Scaled cost penalties must be finite and nonnegative")
    return penalties


def _candidate(scores, probabilities, costs, weight, cost_scale="minmax"):
    if weight == 0:
        return max(range(len(scores)), key=lambda index: (scores[index], -index))
    normalized = scaled_costs(costs, cost_scale)
    penalties = tuple(weight * value for value in normalized)
    if cost_scale != "minmax" and any(not math.isfinite(value) for value in penalties):
        raise ValueError("Weighted cost penalties must be finite")
    def rank(index):
        return probabilities[index] - penalties[index], probabilities[index], -costs[index], -index
    return max(range(len(scores)), key=rank)


def select_candidate(*, scores, category, models, actual_costs=None, cost_mode="actual",
                     cost_profile=None, cost_weight=0., task_policy=None,
                     task_policy_enabled=False, guard_enabled=False, cost_scale="minmax",
                     cost_scale_benches=None):
    names, benchmark = _models(models), _benchmark(category)
    values = _vector(scores, len(names), "scores")
    weight = _number(cost_weight, "cost_weight", nonnegative=True)
    if type(task_policy_enabled) is not bool or type(guard_enabled) is not bool:
        raise ValueError("task_policy_enabled and guard_enabled must be boolean")
    if guard_enabled and not task_policy_enabled:
        raise ValueError("guard_enabled requires task_policy_enabled")
    if cost_mode not in ("actual", "train_task_mean"):
        raise ValueError("cost_mode must be 'actual' or 'train_task_mean'")
    scope = _scale_scope(cost_scale, cost_mode, cost_scale_benches)
    selected_scale = cost_scale if scope is None or benchmark in scope else "minmax"
    anchor, threshold, only = None, None, False
    if task_policy_enabled:
        weight, anchor, threshold, only = _spec(task_policy, benchmark, len(names))
    costs = None
    if cost_mode == "train_task_mean" or weight > 0:
        costs = selection_costs(category=category, models=names, actual_costs=actual_costs,
                                cost_mode=cost_mode, cost_profile=cost_profile)
    if guard_enabled and only:
        return anchor
    probabilities = tuple(_sigmoid(value) for value in values)
    candidate = _candidate(values, probabilities, costs, weight, selected_scale)
    if (guard_enabled and threshold is not None and candidate != anchor
            and not probabilities[candidate] - probabilities[anchor] > threshold):
        return anchor
    return candidate
