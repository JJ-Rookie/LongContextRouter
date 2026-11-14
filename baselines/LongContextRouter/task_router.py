"""Input-only task prediction and hierarchical cost-aware model selection.

This module is opt-in and deliberately independent of the historical inference
entrypoints.  The selector never accepts a true benchmark/category argument;
categories, labels, and realized costs are consumed only by offline reporting.
"""
from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from . import causal_cost as cost


DEFAULT_TASKS = ("bbh", "memory_agent_bench", "mmau", "swe_bench_pro")


def _names(values, label):
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{label} must be a nonempty ordered sequence of unique names")
    return tuple(values)


def _number(value, label, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return result


class TaskRouterHead(torch.nn.Module):
    """Small task classifier trained over a frozen prompt representation."""

    def __init__(self, input_dim: int, tasks: Sequence[str]):
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self.tasks = _names(tasks, "tasks")
        self.input_dim = input_dim
        self.classifier = torch.nn.Linear(input_dim, len(self.tasks))

    def forward(self, features):
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.size(1) != self.input_dim
                or not bool(torch.isfinite(features).all())):
            raise ValueError("features must be a finite [batch, input_dim] tensor")
        return self.classifier(features)


def _classification_metrics(logits, labels, task_count):
    predictions = logits.argmax(dim=1)
    total = int(labels.numel())
    correct = int((predictions == labels).sum().item())
    per_task = []
    for task_id in range(task_count):
        mask = labels == task_id
        count = int(mask.sum().item())
        per_task.append(float((predictions[mask] == task_id).float().mean().item()) if count else 0.)
    return {"correct": correct, "total": total, "accuracy": correct / total,
            "macro_accuracy": math.fsum(per_task) / task_count,
            "per_task_accuracy": per_task}


def _training_tensors(features, labels, input_dim, task_count, label):
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
            or features.size(0) == 0 or features.size(1) != input_dim
            or not bool(torch.isfinite(features).all())):
        raise ValueError(f"{label} features must be finite and nonempty")
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 1
            or labels.numel() != features.size(0) or labels.dtype == torch.bool
            or any(value < 0 or value >= task_count for value in labels.tolist())):
        raise ValueError(f"{label} labels must contain valid task IDs")
    return features.detach().to(device="cpu", dtype=torch.float32).clone(), labels.detach().to(
        device="cpu", dtype=torch.long).clone()


def fit_task_head(train_features, train_labels, validation_features, validation_labels, tasks,
                  *, epochs=200, learning_rate=1e-2, weight_decay=1e-4, seed=42):
    """Fit only a linear head; return immutable CPU state selected on validation."""
    names = _names(tasks, "tasks")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    lr = _number(learning_rate, "learning_rate", minimum=0.)
    decay = _number(weight_decay, "weight_decay", minimum=0.)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(train_features, torch.Tensor) or train_features.ndim != 2:
        raise ValueError("train features must be a rank-2 tensor")
    input_dim = int(train_features.size(1))
    train_x, train_y = _training_tensors(
        train_features, train_labels, input_dim, len(names), "train")
    val_x, val_y = _training_tensors(
        validation_features, validation_labels, input_dim, len(names), "validation")
    counts = torch.bincount(train_y, minlength=len(names))
    if bool((counts == 0).any()):
        raise ValueError("every task requires at least one training example")
    weights = train_y.numel() / (len(names) * counts.to(dtype=torch.float32))
    torch.manual_seed(seed)
    head = TaskRouterHead(input_dim, names)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=decay)
    history, best_key, best_state, best_metrics, best_epoch = [], None, None, None, None
    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = head(train_x)
        train_loss = F.cross_entropy(train_logits, train_y, weight=weights)
        train_loss.backward()
        optimizer.step()
        head.eval()
        with torch.no_grad():
            validation_logits = head(val_x)
            validation_loss = float(F.cross_entropy(validation_logits, val_y).item())
            metrics = _classification_metrics(validation_logits, val_y, len(names))
        row = {"epoch": epoch, "train_loss": float(train_loss.item()),
               "validation_loss": validation_loss, **metrics}
        history.append(row)
        key = (metrics["macro_accuracy"], metrics["accuracy"], -validation_loss, -epoch)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_metrics = key, epoch, copy.deepcopy(row)
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    return {"format_version": 1, "tasks": list(names), "input_dim": input_dim,
            "best_epoch": best_epoch, "best_metrics": best_metrics,
            "class_counts": counts.tolist(), "class_weights": weights.tolist(),
            "training": {"epochs": epochs, "learning_rate": lr,
                         "weight_decay": decay, "seed": seed},
            "history": history, "state_dict": best_state}


def task_head_from_payload(payload):
    if not isinstance(payload, Mapping) or payload.get("format_version") != 1:
        raise ValueError("unsupported task-head payload")
    head = TaskRouterHead(payload.get("input_dim"), payload.get("tasks"))
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("task-head payload is missing state_dict")
    head.load_state_dict(state, strict=True)
    head.eval()
    return head


@torch.no_grad()
def predict_probabilities(head, features, *, temperature=1.):
    value = _number(temperature, "temperature", minimum=0.)
    if value == 0:
        raise ValueError("temperature must be positive")
    probabilities = torch.softmax(head(features.detach().to(device="cpu", dtype=torch.float32)) / value, dim=1)
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("task probabilities must be finite")
    return probabilities


def build_global_profile(task_profile, *, category="all_inputs"):
    """Collapse task means using original training-group counts, never task equality."""
    if not isinstance(category, str) or not category.strip() or "/" in category:
        raise ValueError("global category must be one nonempty root name")
    checked = cost.validate_profile(task_profile, task_profile.get("models"))
    models = checked["models"]
    entries = tuple(checked["by_benchmark"].values())
    counts = [entry["counts"][0] for entry in entries]
    total = sum(counts)
    means = [math.fsum(entry["mean_costs"][model_id] * entry["counts"][model_id]
                       for entry in entries) / total for model_id in range(len(models))]
    provenance = copy.deepcopy(checked["provenance"])
    provenance["aggregation"] = (
        "original-training-group-count-weighted global model means; "
        "benchmark identity unavailable at selection"
    )
    return cost.validate_profile({"schema_version": 1, "mode": "train_task_mean",
        "cost_unit": "whole_problem", "models": models,
        "by_benchmark": {category: {"mean_costs": means, "counts": [total] * len(models)}},
        "provenance": provenance}, models)


def _posterior(values, count):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError("task posterior has the wrong length")
    posterior = tuple(_number(value, "task posterior", minimum=0., maximum=1.) for value in values)
    total = math.fsum(posterior)
    if not math.isclose(total, 1., rel_tol=0., abs_tol=1e-6):
        raise ValueError("task posterior must sum to one")
    return tuple(value / total for value in posterior)


def _task_spec(policy, task, model_count):
    if not isinstance(policy, Mapping) or task not in policy or not isinstance(policy[task], Mapping):
        raise ValueError(f"task policy has no specification for {task!r}")
    spec = policy[task]
    if set(spec) - {"weight", "anchor", "threshold", "anchor_only"} or "weight" not in spec:
        raise ValueError("invalid task policy specification")
    weight = _number(spec["weight"], "task weight", minimum=0.)
    anchor = spec.get("anchor")
    threshold = spec.get("threshold")
    only = spec.get("anchor_only", False)
    if anchor is not None and (type(anchor) is not int or not 0 <= anchor < model_count):
        raise ValueError("anchor is outside the model pool")
    if threshold is not None:
        threshold = _number(threshold, "guard threshold", minimum=0.)
    if threshold is not None and anchor is None:
        raise ValueError("guard threshold requires an anchor")
    if type(only) is not bool or only and anchor is None:
        raise ValueError("anchor_only requires an anchor")
    return weight, anchor, threshold, only


def _sigmoid(value):
    return 1. / (1. + math.exp(-value)) if value >= 0 else math.exp(value) / (1. + math.exp(value))


def _fallback(scores, models, task_profile, weight, scale):
    global_profile = build_global_profile(task_profile)
    return cost.select_candidate(scores=scores, category="all_inputs", models=models,
        actual_costs=None, cost_mode="train_task_mean", cost_profile=global_profile,
        cost_weight=weight, task_policy_enabled=False, guard_enabled=False,
        cost_scale=scale, cost_scale_benches=None)


def route_decision(*, scores, task_posterior, tasks, models, task_profile, task_policy,
                   mode="hard", confidence_threshold=0., fallback_weight=.3,
                   fallback_scale="train_log1p_median", cost_scale="train_log1p_median",
                   cost_scale_tasks=("swe_bench_pro",), guard_enabled=True):
    """Select using predicted task information only; no true category is accepted."""
    task_names, model_names = _names(tasks, "tasks"), _names(models, "models")
    values = tuple(_number(value, "score") for value in scores)
    if len(values) != len(model_names):
        raise ValueError("scores must match the model count")
    posterior = _posterior(task_posterior, len(task_names))
    threshold = _number(confidence_threshold, "confidence_threshold", minimum=0., maximum=1.)
    fallback = _number(fallback_weight, "fallback_weight", minimum=0.)
    if type(guard_enabled) is not bool:
        raise ValueError("guard_enabled must be boolean")
    if mode not in {"hard", "soft"}:
        raise ValueError("mode must be 'hard' or 'soft'")
    profile = cost.validate_profile(task_profile, model_names)
    if set(profile["by_benchmark"]) != set(task_names):
        raise ValueError("task profile and task order must name the same tasks")
    scope = tuple(cost_scale_tasks)
    if len(set(scope)) != len(scope) or any(task not in task_names for task in scope):
        raise ValueError("cost_scale_tasks must be unique known tasks")
    predicted_id = max(range(len(task_names)), key=lambda index: (posterior[index], -index))
    predicted_task, confidence = task_names[predicted_id], posterior[predicted_id]
    if confidence < threshold:
        selected = _fallback(values, model_names, profile, fallback, fallback_scale)
        return {"selected_model_id": selected, "predicted_task": predicted_task,
                "task_confidence": confidence, "used_fallback": True}
    for task in task_names:
        _task_spec(task_policy, task, len(model_names))
    if mode == "hard":
        selected = cost.select_candidate(scores=values, category=predicted_task, models=model_names,
            actual_costs=None, cost_mode="train_task_mean", cost_profile=profile,
            task_policy=task_policy, task_policy_enabled=True, guard_enabled=guard_enabled,
            cost_scale=cost_scale, cost_scale_benches=list(scope) or None)
    else:
        probabilities = tuple(_sigmoid(value) for value in values)
        penalties = [0.] * len(model_names)
        expected_costs = [0.] * len(model_names)
        for task_id, task in enumerate(task_names):
            weight, _anchor, _guard, _only = _task_spec(task_policy, task, len(model_names))
            task_costs = cost.estimate_costs(profile, task)
            scale = cost_scale if task in scope else "minmax"
            scaled = cost.scaled_costs(task_costs, scale)
            for model_id in range(len(model_names)):
                penalties[model_id] += posterior[task_id] * weight * scaled[model_id]
                expected_costs[model_id] += posterior[task_id] * task_costs[model_id]
        selected = max(range(len(model_names)), key=lambda index: (
            probabilities[index] - penalties[index], probabilities[index],
            -expected_costs[index], -index))
        weight, anchor, guard, only = _task_spec(task_policy, predicted_task, len(model_names))
        if guard_enabled and only:
            selected = anchor
        elif (guard_enabled and anchor is not None and guard is not None and selected != anchor
              and not probabilities[selected] - probabilities[anchor] > guard):
            selected = anchor
    return {"selected_model_id": selected, "predicted_task": predicted_task,
            "task_confidence": confidence, "used_fallback": False}


def select_candidate(**kwargs):
    return route_decision(**kwargs)["selected_model_id"]


def _summary(choices, models):
    success = sum(choice["success"] for choice in choices)
    actual = math.fsum(choice["actual_cost"] for choice in choices)
    return {"total_prompts": len(choices), "total_successes": success,
            "sample_accuracy": success / len(choices), "total_cost": actual,
            "total_actual_cost": actual, "mean_selected_cost": actual / len(choices),
            "model_counts": dict(Counter(models[choice["selected_model_id"]] for choice in choices)),
            "predicted_task_counts": dict(Counter(choice["predicted_task"] for choice in choices)),
            "fallback_count": sum(choice["used_fallback"] for choice in choices)}


def evaluate_records(records, *, task_posteriors, models, **selection_options):
    """Route first, then use true labels/cost/category only for offline accounting."""
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("nonempty records are required")
    model_names = _names(models, "models")
    identifiers = [record.get("prompt_id") for record in records]
    if (any(not isinstance(identifier, str) or not identifier for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError("records require unique nonempty prompt IDs")
    if not isinstance(task_posteriors, Mapping) or set(task_posteriors) != set(identifiers):
        raise ValueError("task posteriors must exactly match record prompt IDs")
    decisions = [route_decision(scores=record["scores"],
        task_posterior=task_posteriors[record["prompt_id"]], models=model_names,
        **selection_options) for record in records]
    choices = []
    for record, decision in zip(records, decisions):
        model_id = decision["selected_model_id"]
        labels, costs = record["labels"], record["costs"]
        if len(labels) != len(model_names) or len(costs) != len(model_names):
            raise ValueError("record outcomes must match the model count")
        category = record.get("category")
        if not isinstance(category, str) or not category.split("/", 1)[0]:
            raise ValueError("reporting category must be nonempty")
        choices.append({"prompt_id": record["prompt_id"], "category": category,
            **decision, "success": int(labels[model_id]), "cost": float(costs[model_id]),
            "actual_cost": float(costs[model_id]),
            "task_posterior": list(task_posteriors[record["prompt_id"]])})
    benches = {bench: _summary([choice for choice in choices
               if choice["category"].split("/", 1)[0] == bench], model_names)
               for bench in sorted({choice["category"].split("/", 1)[0] for choice in choices})}
    return {**_summary(choices, model_names), "by_benchmark": benches,
            "benchmark_macro_accuracy": math.fsum(value["sample_accuracy"]
                for value in benches.values()) / len(benches),
            "cost_by_benchmark": {key: value["total_cost"] for key, value in benches.items()},
            "choices": choices}
