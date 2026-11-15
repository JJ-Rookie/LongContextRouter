from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from baselines.SWERouter5.data import MODEL_ORDER, load_examples
from baselines.SWERouter5.policy import (
    estimate_remaining_costs, evaluate_policy, select_validation_policy,
)
from baselines.SWERouter5.value_head import MODEL_ID, MODEL_REVISION


EXPLORER_INDEX = MODEL_ORDER.index("deepseek-v4-flash-0731")
DEFAULT_WEIGHTS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0,
                   30.0, 100.0)
DEFAULT_PREPARED = Path("data/swe_router/prepared")
DEFAULT_SWE_ROOT = Path("data/swe_agent")


def audit_examples(examples: Mapping[str, Sequence[Mapping]], *,
                   require_exact: bool = False) -> dict:
    report = {"splits": {}, "all_exact": True}
    for split in ("train", "validation", "test"):
        rows = examples.get(split)
        if not rows:
            raise ValueError(f"{split} split is empty")
        reasons = Counter()
        missing = []
        for row in rows:
            cost, status = row.get("prefix_cost"), row.get("prefix_cost_status")
            full = row["full_costs"][EXPLORER_INDEX]
            valid = (status == "exact" and isinstance(cost, (int, float))
                     and not isinstance(cost, bool) and math.isfinite(cost)
                     and 0 <= cost <= full + 1e-9)
            if not valid:
                missing.append(row["instance_id"])
                reasons[row.get("prefix_cost_reason") or str(status)] += 1
        report["splits"][split] = {
            "count": len(rows), "exact_count": len(rows) - len(missing),
            "unavailable_count": len(missing), "unavailable_ids": missing,
            "reasons": dict(reasons),
        }
        if missing:
            report["all_exact"] = False
            if require_exact:
                raise ValueError(f"{split}: {len(missing)} prefix costs are unavailable; "
                                 "see audit output before training")
    text_ready = all(
        isinstance(row.get("problem"), str) and row["problem"].strip()
        and isinstance(row.get("prefix_text"), str) and row["prefix_text"].strip()
        for split in ("train", "validation", "test") for row in examples[split]
    )
    report["fit_ready"] = bool(
        text_ready and report["splits"]["train"]["exact_count"] > 0
        and all(report["splits"][split]["unavailable_count"] == 0
                for split in ("validation", "test"))
    )
    return report


def validation_budget(rows: Sequence[Mapping], *, explorer_index: int,
                      multiplier: float) -> float:
    if not rows or not isinstance(multiplier, (float, int)) or isinstance(multiplier, bool):
        raise ValueError("validation rows and a positive budget multiplier are required")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("budget multiplier must be finite and positive")
    if not 0 <= explorer_index < len(rows[0]["full_costs"]):
        raise ValueError("explorer index is outside the model pool")
    baseline = sum(float(row["full_costs"][explorer_index]) for row in rows)
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("fixed-explorer validation cost is invalid")
    return baseline * multiplier


def _write_json(path: Path, payload: Mapping) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _artifact_provenance(examples: Mapping[str, Sequence[Mapping]]) -> dict:
    return {split: [
        {"prompt_id": row["prompt_id"], "instance_id": row["instance_id"]}
        for row in examples[split]
    ] for split in ("train", "validation", "test")}


def _value_input_summary(value_inputs: Mapping[str, Sequence[Mapping]]) -> dict:
    return {split: {
        "count": len(rows),
        "max_input_tokens": max(row["input_tokens"] for row in rows),
        "compressed_rows": sum(row["truncated_observations"] > 0 for row in rows),
    } for split, rows in value_inputs.items()}


def _source_manifest(prepared_dir: Path, swe_root: Path, args) -> dict:
    return {
        "protocol": "SWE-Router five-model method extension, SWE-bench Pro only",
        "paper": "https://arxiv.org/html/2607.00053",
        "implementation": "Local method extension; not the official implementation.",
        "model_order": list(MODEL_ORDER), "explorer_model": MODEL_ORDER[EXPLORER_INDEX],
        "exploration_steps": args.k, "prepared_dir": str(prepared_dir.resolve()),
        "swe_root": str(swe_root.resolve()), "validation_budget_multiplier": args.budget_multiplier,
        "candidate_cost_weights": list(DEFAULT_WEIGHTS),
        "value_head": {"base_model": args.base_model, "epochs": args.epochs,
                       "base_model_revision": MODEL_REVISION if args.base_model == MODEL_ID else None,
                       "max_length": args.max_length, "seed": args.seed,
                       "learning_rate": args.learning_rate,
                       "warmup_steps": args.warmup_steps,
                       "gradient_accumulation": args.gradient_accumulation},
        "input_representation": "complete task/thought/action; observations only may "
                                "use deterministic head-tail excerpts with explicit omission markers",
        "cost_accounting": "native-cache-repriced Flash prefix plus frozen selected full run; "
                           "continuation uses frozen full Flash run",
    }


def fit(args, examples: Mapping[str, Sequence[Mapping]]) -> dict:
    from baselines.SWERouter5.value_head import (
        predict_from_adapter, preflight_value_inputs, train_value_head,
        validate_examples,
    )

    audit = audit_examples(examples)
    if not audit["fit_ready"]:
        raise ValueError("audit is not fit-ready: verify text and exact validation/test costs")
    for split in ("validation", "test"):
        if audit["splits"][split]["unavailable_count"]:
            raise ValueError(f"{split}: exact prefix billing is required for every row")
    cost_prior_rows = [row for row in examples["train"]
                       if row["prefix_cost_status"] == "exact"]
    if not cost_prior_rows:
        raise ValueError("train: no exact prefix costs for remaining-cost priors")
    validate_examples(examples["train"], examples["validation"])
    artifacts = _artifact_provenance(examples)
    value_inputs = preflight_value_inputs(
        examples, base_model=args.base_model, max_length=args.max_length)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.incomplete-",
                                    dir=output_dir.parent))
    prepared_dir, swe_root = Path(args.prepared_dir), Path(args.swe_root)
    artifact_path = staging / "input_artifacts.json"
    view_path = staging / "value_inputs.json"
    _write_json(artifact_path, artifacts)
    _write_json(view_path, value_inputs)
    _write_json(staging / "run_manifest.json", {
        **_source_manifest(prepared_dir, swe_root, args),
        "cost_prior_training_count": len(cost_prior_rows),
    })
    _write_json(staging / "audit.json", audit)
    training = train_value_head(
        examples["train"], examples["validation"], staging / "training",
        base_model=args.base_model, max_length=args.max_length, epochs=args.epochs,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        gradient_accumulation=args.gradient_accumulation, seed=args.seed,
        device=args.device,
    )
    remaining = estimate_remaining_costs(cost_prior_rows, explorer_index=EXPLORER_INDEX)
    adapter = Path(training["adapter_dir"])
    validation_predictions = predict_from_adapter(
        examples["validation"], adapter, base_model=args.base_model,
        max_length=args.max_length, device=args.device)
    budget = validation_budget(examples["validation"], explorer_index=EXPLORER_INDEX,
                               multiplier=args.budget_multiplier)
    policy = select_validation_policy(
        examples["validation"], validation_predictions, remaining,
        weights=DEFAULT_WEIGHTS, max_cost_usd=budget, explorer_index=EXPLORER_INDEX)
    _write_json(staging / "validation_policy.json", {
        "budget_usd": budget, "remaining_cost_estimates_usd": remaining,
        "cost_weight": policy["cost_weight"],
        "candidate_summary": policy["candidate_summary"],
        "validation": policy["validation"],
        "training": {**training, "adapter_dir": "training/adapter"},
    })
    test_predictions = predict_from_adapter(
        examples["test"], adapter, base_model=args.base_model,
        max_length=args.max_length, device=args.device)
    evaluated = evaluate_policy(examples["test"], test_predictions, remaining,
                                policy["cost_weight"], explorer_index=EXPLORER_INDEX)
    decisions = [{**decision, "predicted_success": prediction,
                  "selected_model": MODEL_ORDER[decision["selected_model_id"]]}
                 for decision, prediction in zip(evaluated["decisions"], test_predictions)]
    test = {**evaluated, "decisions": decisions,
            "selection_counts": dict(Counter(
                decision["selected_model"] for decision in decisions))}
    _write_json(staging / "test_result.json", test)
    if output_dir.exists():
        raise FileExistsError(f"output directory appeared during training: {output_dir}")
    os.rename(staging, output_dir)
    return {"validation": policy["validation"], "test": {
        key: value for key, value in test.items() if key != "decisions"},
        "output_dir": str(output_dir)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit or train the five-model SWE router.")
    parser.add_argument("command", choices=("audit", "fit"))
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--swe-root", type=Path, default=DEFAULT_SWE_ROOT)
    parser.add_argument("--k", type=int, default=3, help="Flash exploratory actions")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budget-multiplier", type=float, default=1.0)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "fit" and args.output_dir is None:
        parser.error("fit requires --output-dir")
    examples = load_examples(args.prepared_dir, args.swe_root, args.k)
    if args.command == "audit":
        report = audit_examples(examples)
        value_inputs = None
        if report["fit_ready"]:
            from baselines.SWERouter5.value_head import preflight_value_inputs

            try:
                value_inputs = preflight_value_inputs(
                    examples, base_model=args.base_model, max_length=args.max_length)
                report = {**report, "value_input_summary": _value_input_summary(value_inputs)}
            except ValueError as exc:
                report = {**report, "fit_ready": False,
                          "tokenizer_preflight_error": str(exc)}
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            artifact_path = args.output_dir / "input_artifacts.json"
            _write_json(artifact_path, _artifact_provenance(examples))
            if value_inputs is not None:
                view_path = args.output_dir / "value_inputs.json"
                _write_json(view_path, value_inputs)
            _write_json(args.output_dir / "audit.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["fit_ready"] else 2
    result = fit(args, examples)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
