from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from baselines.SWERouter5.data import MODEL_ORDER
from baselines.SWERouter5.four_bench_data import (
    FOUR_BENCH_COUNTS, TEST_BENCH_COUNTS, load_four_bench_examples,
)
from baselines.SWERouter5.four_bench_policy import (
    BENCHES, estimate_remaining_costs_by_bench, evaluate_four_bench_policy,
    select_four_bench_validation_policy,
)
from baselines.SWERouter5.run import (
    DEFAULT_PREPARED, DEFAULT_SWE_ROOT, DEFAULT_WEIGHTS, EXPLORER_INDEX,
    _value_input_summary, _write_json,
)
from baselines.SWERouter5.value_head import MODEL_ID, MODEL_REVISION


PROTOCOL = ("Hybrid complete-response cascade on BBH, MemoryAgentBench, MMAU; "
            "K3 agent prefix continuation/restart on SWE-bench Pro")
BENCH_COUNTS = {
    "train": {"bbh": 392, "memory_agent_bench": 155, "mmau": 160,
              "swe_bench_pro": 160},
    "validation": {"bbh": 108, "memory_agent_bench": 45, "mmau": 40,
                   "swe_bench_pro": 40},
    "test": TEST_BENCH_COUNTS,
}


def _valid_cost(value: object, full: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0
            and isinstance(full, (int, float)) and not isinstance(full, bool)
            and math.isfinite(full) and full >= 0 and value <= full + 1e-9)


def _check_row(row: Mapping, split: str, explorer_index: int = EXPLORER_INDEX) -> bool:
    bench = row.get("bench")
    if bench not in BENCHES or row.get("split") != split:
        raise ValueError(f"{split}: missing benchmark identity or split tag")
    pid = row.get("prompt_id")
    iid, key = row.get("instance_id"), row.get("prompt_key")
    if (type(pid) is not int or pid < 0 or not isinstance(iid, str) or not iid
            or key != f"{bench}:{iid}"):
        raise ValueError(f"{split}: invalid prompt ID or prompt key")
    if any(not isinstance(row.get(name), str) or not row[name].strip()
           for name in ("problem", "prefix_text")):
        raise ValueError(f"{split} {key}: task or observed Flash text is empty")
    labels, costs = row.get("labels"), row.get("full_costs")
    if (not isinstance(labels, (list, tuple)) or len(labels) != len(MODEL_ORDER)
            or any(type(value) is not int or value not in (0, 1) for value in labels)
            or not isinstance(costs, (list, tuple)) or len(costs) != len(MODEL_ORDER)
            or any(not _valid_cost(value, value) for value in costs)):
        raise ValueError(f"{split} {key}: incomplete frozen five-model labels or costs")
    if bench == "swe_bench_pro":
        if row.get("probe_kind") != "agent_prefix":
            raise ValueError(f"{split} {key}: SWE probe must be an agent prefix")
    else:
        if (row.get("probe_kind") != "complete_answer"
                or row.get("prefix_cost_source") != "frozen_full_run"
                or row.get("early_finish") is not False):
            raise ValueError(f"{split} {key}: non-SWE probe is not a complete Flash run")
    exact = row.get("prefix_cost_status") == "exact"
    if exact:
        prefix = row.get("prefix_cost")
        if not _valid_cost(prefix, costs[explorer_index]):
            raise ValueError(f"{split} {key}: exact probe cost is invalid")
        if bench != "swe_bench_pro" and not math.isclose(
            prefix, costs[explorer_index], rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError(f"{split} {key}: complete Flash probe cost differs from frozen full run")
    elif row.get("prefix_cost") is not None:
        raise ValueError(f"{split} {key}: unavailable probe cost has a numeric value")
    return exact


def audit_four_bench_examples(examples: Mapping[str, Sequence[Mapping]], *,
                              require_ready: bool = False,
                              explorer_index: int = EXPLORER_INDEX) -> dict:
    if set(examples) != set(FOUR_BENCH_COUNTS):
        raise ValueError("four-benchmark split names differ from frozen protocol")
    report = {"protocol": PROTOCOL, "splits": {}, "all_exact": True}
    ids, keys = set(), set()
    for split in ("train", "validation", "test"):
        rows = examples[split]
        expected = FOUR_BENCH_COUNTS[split]
        if not isinstance(rows, (list, tuple)) or len(rows) != expected:
            raise ValueError(f"{split}: expected {expected} frozen prompts")
        counts, sources, unavailable = Counter(), Counter(), []
        for row in rows:
            exact = _check_row(row, split, explorer_index)
            pid, key = row["prompt_id"], row["prompt_key"]
            if pid in ids:
                raise ValueError(f"duplicate prompt ID: {pid}")
            if key in keys:
                raise ValueError(f"duplicate prompt key: {key}")
            ids.add(pid)
            keys.add(key)
            counts[row["bench"]] += 1
            sources[row.get("prefix_cost_source") or "native_trace"] += 1
            if not exact:
                unavailable.append({"prompt_id": pid, "prompt_key": key,
                                    "reason": row.get("prefix_cost_reason") or
                                    str(row.get("prefix_cost_status"))})
        if counts != BENCH_COUNTS[split]:
            raise ValueError(f"{split}: benchmark counts differ from frozen comparison")
        report["splits"][split] = {
            "count": len(rows), "per_bench": dict(counts),
            "exact_count": len(rows) - len(unavailable),
            "unavailable_count": len(unavailable), "unavailable": unavailable,
            "cost_sources": dict(sources),
        }
        if unavailable:
            report["all_exact"] = False
    train_unavailable = [row for row in examples["train"]
                         if row.get("prefix_cost_status") != "exact"]
    train_allowed = (len(train_unavailable) <= 1 and all(
        row["bench"] == "swe_bench_pro" for row in train_unavailable))
    prior_rows = [row for row in examples["train"]
                  if row.get("prefix_cost_status") == "exact"]
    report["train_cost_prior_count"] = len(prior_rows)
    report["fit_ready"] = bool(
        train_allowed and all(any(row["bench"] == bench for row in prior_rows)
                              for bench in BENCHES)
        and all(report["splits"][split]["unavailable_count"] == 0
                for split in ("validation", "test")))
    if require_ready and not report["fit_ready"]:
        raise ValueError("four-benchmark audit is not fit-ready: validation/test "
                         "require exact costs and train allows at most one unavailable SWE cost")
    return report


def _artifact_provenance(examples: Mapping[str, Sequence[Mapping]]) -> dict:
    return {split: [
        {"prompt_id": row["prompt_id"], "prompt_key": row["prompt_key"],
         "bench": row["bench"], "probe_kind": row["probe_kind"],
         "probe_cost_source": row.get("prefix_cost_source") or "native_trace"}
        for row in examples[split]
    ] for split in ("train", "validation", "test")}


def _prediction_records(rows: Sequence[Mapping], predictions: Sequence[Sequence[float]],
                        value_inputs: Sequence[Mapping]) -> list[dict]:
    if len(rows) != len(predictions) or len(rows) != len(value_inputs):
        raise ValueError("predictions, examples and value inputs are misaligned")
    result = []
    for row, prediction, view in zip(rows, predictions, value_inputs):
        if (view.get("prompt_id") != row["prompt_id"] or
                view.get("instance_id") != row["instance_id"]):
            raise ValueError("value-input provenance differs from prediction row")
        if (len(prediction) != len(MODEL_ORDER) or
                any(not isinstance(p, (int, float)) or isinstance(p, bool)
                    or not math.isfinite(p) or not 0 <= p <= 1 for p in prediction)):
            raise ValueError("five finite probability predictions are required")
        result.append({"prompt_id": row["prompt_id"], "prompt_key": row["prompt_key"],
                       "bench": row["bench"],
                       "predicted_success": list(prediction)})
    return result


def _source_manifest(prepared_dir: Path, swe_root: Path, args) -> dict:
    return {
        "protocol": PROTOCOL, "method_status": "four-benchmark hybrid extension; not official SWE-Router",
        "paper": "https://arxiv.org/html/2607.00053",
        "model_order": list(MODEL_ORDER),
        "explorer_model": getattr(args, "explorer_model", MODEL_ORDER[EXPLORER_INDEX]),
        "exploration_steps_swe": args.k,
        "non_swe_probe": "complete recorded explorer response; charged frozen full explorer cost",
        "selected_alternative": "restart from original task; add probe cost to selected full-run cost",
        "prepared_dir": str(prepared_dir.resolve()),
        "swe_root": str(swe_root.resolve()),
        "validation_budget_multiplier": args.budget_multiplier,
        "candidate_cost_weights": list(DEFAULT_WEIGHTS),
        "value_head": {"base_model": args.base_model, "epochs": args.epochs,
                       "base_model_revision": MODEL_REVISION if args.base_model == MODEL_ID else None,
                       "precision": "bf16_autocast", "attention": "sdpa",
                       "max_length": args.max_length, "seed": args.seed,
                       "learning_rate": args.learning_rate,
                       "warmup_steps": args.warmup_steps,
                       "gradient_accumulation": args.gradient_accumulation},
        "input_representation": "task plus observed explorer trace or recorded complete response; "
                                "SWE observations and oversized MMAU prior tool observations "
                                "may be deterministically head-tail excerpted; input token counts "
                                "and excerpt settings are recorded in value_inputs.json",
        "cost_accounting": "SWE native-cache-repriced prefix or non-SWE frozen full explorer "
                           "probe, plus frozen selected full run on restart",
    }


def fit(args, examples: Mapping[str, Sequence[Mapping]]) -> dict:
    from baselines.SWERouter5.value_head import (
        predict_from_adapter, preflight_value_inputs, train_value_head,
        validate_examples,
    )

    if args.k != 3:
        raise ValueError("four-benchmark hybrid comparison is pinned to K=3 SWE actions")
    explorer = getattr(args, "explorer_model", MODEL_ORDER[EXPLORER_INDEX])
    if explorer not in ("deepseek-v4-flash-0731", "deepseek-v4-pro-0813"):
        raise ValueError("unsupported four-benchmark explorer model")
    explorer_index = MODEL_ORDER.index(explorer)
    audit = audit_four_bench_examples(
        examples, require_ready=True, explorer_index=explorer_index)
    validate_examples(examples["train"], examples["validation"])
    artifacts = _artifact_provenance(examples)
    value_inputs = preflight_value_inputs(
        examples, base_model=args.base_model, max_length=args.max_length)
    for split in ("train", "validation", "test"):
        if len(value_inputs[split]) != len(examples[split]):
            raise ValueError(f"{split}: tokenizer preflight omitted an input")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.incomplete-",
                                    dir=output_dir.parent))
    artifact_path, view_path = staging / "input_artifacts.json", staging / "value_inputs.json"
    _write_json(artifact_path, artifacts)
    _write_json(view_path, value_inputs)
    prior_rows = [row for row in examples["train"]
                  if row["prefix_cost_status"] == "exact"]
    _write_json(staging / "run_manifest.json", {
        **_source_manifest(Path(args.prepared_dir), Path(args.swe_root), args),
        "train_count": len(examples["train"]),
        "cost_prior_training_count": len(prior_rows),
    })
    _write_json(staging / "audit.json", {
        **audit, "value_input_summary": _value_input_summary(value_inputs)})
    training = train_value_head(
        examples["train"], examples["validation"], staging / "training",
        base_model=args.base_model, max_length=args.max_length, epochs=args.epochs,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        gradient_accumulation=args.gradient_accumulation, seed=args.seed,
        device=args.device,
    )
    priors = estimate_remaining_costs_by_bench(prior_rows,
                                               explorer_index=explorer_index)
    adapter = Path(training["adapter_dir"])
    val_predictions = predict_from_adapter(
        examples["validation"], adapter, base_model=args.base_model,
        max_length=args.max_length, device=args.device)
    val_records = _prediction_records(examples["validation"], val_predictions,
                                      value_inputs["validation"])
    policy = select_four_bench_validation_policy(
        examples["validation"], val_predictions, priors,
        weights=DEFAULT_WEIGHTS, budget_multiplier=args.budget_multiplier,
        explorer_index=explorer_index)
    _write_json(staging / "validation_predictions.json", val_records)
    _write_json(staging / "validation_policy.json", {
        **policy, "remaining_cost_estimates_usd": priors,
        "training": {**training, "adapter_dir": "training/adapter"},
    })
    test_predictions = predict_from_adapter(
        examples["test"], adapter, base_model=args.base_model,
        max_length=args.max_length, device=args.device)
    test_records = _prediction_records(examples["test"], test_predictions,
                                       value_inputs["test"])
    _write_json(staging / "test_predictions.json", test_records)
    evaluated = evaluate_four_bench_policy(
        examples["test"], test_predictions, priors, policy["cost_weight"],
        explorer_index=explorer_index)
    decisions = [{**decision, "selected_model": MODEL_ORDER[decision["selected_model_id"]],
                  "predicted_success": prediction}
                 for decision, prediction in zip(evaluated["decisions"], test_predictions)]
    test = {**evaluated, "decisions": decisions,
            "selection_counts": dict(Counter(item["selected_model"] for item in decisions))}
    _write_json(staging / "test_result.json", test)
    if output_dir.exists():
        raise FileExistsError(f"output directory appeared during training: {output_dir}")
    os.rename(staging, output_dir)
    return {"validation": {key: value for key, value in policy["validation"].items()
                           if key != "decisions"},
            "test": {key: value for key, value in test.items() if key != "decisions"},
            "output_dir": str(output_dir)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit or train the four-benchmark SWE router.")
    parser.add_argument("command", choices=("audit", "fit"))
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--swe-root", type=Path, default=DEFAULT_SWE_ROOT)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--explorer-model", default=MODEL_ORDER[EXPLORER_INDEX],
                        choices=("deepseek-v4-flash-0731", "deepseek-v4-pro-0813"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budget-multiplier", type=float, default=1.0)
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.k != 3:
        parser.error("four-benchmark hybrid comparison is pinned to --k 3")
    if args.command == "fit" and args.output_dir is None:
        parser.error("fit requires --output-dir")
    examples = load_four_bench_examples(args.prepared_dir, args.swe_root, args.k,
                                        args.explorer_model)
    if args.command == "audit":
        report = audit_four_bench_examples(
            examples, explorer_index=MODEL_ORDER.index(args.explorer_model))
        if report["fit_ready"]:
            from baselines.SWERouter5.value_head import preflight_value_inputs
            try:
                value_inputs = preflight_value_inputs(
                    examples, base_model=args.base_model, max_length=args.max_length)
                report["value_input_summary"] = _value_input_summary(value_inputs)
            except (OSError, RuntimeError, ValueError) as exc:
                report["fit_ready"] = False
                report["tokenizer_preflight_error"] = str(exc)
        if args.output_dir is not None:
            if args.output_dir.exists():
                raise FileExistsError(f"audit output already exists: {args.output_dir}")
            args.output_dir.mkdir(parents=True)
            _write_json(args.output_dir / "audit.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["fit_ready"] else 2
    result = fit(args, examples)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
