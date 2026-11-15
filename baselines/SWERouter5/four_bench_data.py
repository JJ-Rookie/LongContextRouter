from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from baselines.SWERouter5.cost import actionable_indices
from baselines.SWERouter5.data import (
    MODEL_ORDER, PREPARED_COUNTS, _prepared, _selected, _v7_split,
    load_examples,
)


FOUR_BENCH_COUNTS = {"train": 867, "validation": 233, "test": 382}
SOURCE_BENCH_COUNTS = {"bbh": 500, "memory_agent_bench": 200,
                       "mmau": 200, "swe_bench_pro": 200}
TEST_BENCH_COUNTS = {"bbh": 100, "memory_agent_bench": 100,
                     "mmau": 82, "swe_bench_pro": 100}
EXPLORER = "deepseek-v4-flash-0731"
NON_SWE = frozenset({"bbh", "memory_agent_bench", "mmau"})


def read_complete_probe(base: Path, instance_id: str, bench: str) -> dict:
    if (bench not in NON_SWE or not isinstance(instance_id, str)
            or not instance_id or Path(instance_id).name != instance_id):
        raise ValueError("invalid benchmark or artifact instance ID")
    base = Path(base)
    if base.name != instance_id:
        raise ValueError("artifact directory differs from instance ID")
    traj_path = base / f"{instance_id}.traj"
    pred_path = base / f"{instance_id}.pred"
    trace_path = base / f"{instance_id}.trace.log"
    for path in (traj_path, pred_path, trace_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing Flash probe artifact: {path}")
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(pred, dict):
        raise ValueError("Flash probe artifact root is not an object")
    info, steps = data.get("info"), data.get("trajectory")
    if (not isinstance(info, dict) or not isinstance(steps, list)
            or info.get("instance_id") != instance_id
            or pred.get("instance_id") != instance_id):
        raise ValueError("Flash probe instance ID or trajectory is invalid")
    if info.get("exit_status") != "submitted":
        raise ValueError("Flash probe was not submitted")
    stats = info.get("model_stats")
    if (not isinstance(stats, dict) or type(stats.get("api_calls")) is not int
            or stats["api_calls"] < 1 or len(steps) != stats["api_calls"]):
        raise ValueError("Flash probe API-call count differs from trajectory")
    if pred.get("model_name_or_path") != base.parent.name:
        raise ValueError("Flash prediction source split differs from trajectory")
    output = pred.get("model_patch")
    if (not isinstance(output, str) or not output.strip()
            or output != info.get("submission")):
        raise ValueError("Flash prediction is empty or differs from submission")
    actions = actionable_indices(steps)
    if len(actions) != 1 or actions[0] != len(steps) - 1:
        raise ValueError("Flash probe lacks one final actionable submission")
    final = steps[actions[0]]
    channel = "response" if bench == "mmau" else "observation"
    if final.get(channel) != output:
        raise ValueError("Flash prediction differs from final model response")
    if bench == "mmau":
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ValueError("MMAU Flash response is not JSON") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("choices"), list):
            raise ValueError("MMAU Flash response has no choices")
    return {
        "prefix_text": output,
        "probe_kind": "complete_answer",
        "probe_api_calls": stats["api_calls"],
    }


def _v7_all_keys(selected: dict[int, dict], swe_root: Path) -> tuple[set[str], set[str]]:
    from .source_utils import split_prompt_keys

    source = {row["prompt_key"]: row["category"] for pid, row in selected.items()
              if pid < PREPARED_COUNTS["train"] + PREPARED_COUNTS["validation"]}
    if len(source) != 1100 or Counter(key.split(":", 1)[0] for key in source) != SOURCE_BENCH_COUNTS:
        raise ValueError("selected v7 source pool has wrong benchmark coverage")
    with (swe_root / "data/benchmark_splits/ifbench/train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            iid = str(row.get("key", row.get("id", "")))
            labels = row.get("instruction_id_list") or []
            tier = str(labels[0]).split(":", 1)[0] if labels else "unknown"
            key = f"ifbench:{iid}"
            if not iid or key in source:
                raise ValueError("invalid or duplicate IFBench key")
            source[key] = f"ifbench/{tier}"
    if len(source) != 1300:
        raise ValueError("v7 source pool must contain 1300 prompt keys")
    items = {key: SimpleNamespace(category=value) for key, value in source.items()}
    train, validation = split_prompt_keys(sorted(source), items, 0.8, 43)
    if (len(train) != 1024 or len(validation) != 276
            or train | validation != set(source) or train & validation):
        raise ValueError("v7 train/validation split does not partition the source pool")
    swe_train, swe_validation = _v7_split(selected, swe_root)
    if ({key for key in train if key.startswith("swe_bench_pro:")} != swe_train
            or {key for key in validation if key.startswith("swe_bench_pro:")} != swe_validation):
        raise ValueError("v7 SWE split differs from the reconstructed SWE-only split")
    return train, validation


def load_four_bench_examples(prepared_dir: Path, swe_root: Path, k: int,
                             explorer: str = EXPLORER) -> dict[str, list[dict]]:
    if explorer not in (EXPLORER, "deepseek-v4-pro-0813"):
        raise ValueError("unsupported four-benchmark explorer model")
    prepared_dir, swe_root = Path(prepared_dir), Path(swe_root)
    manifest, records, questions = _prepared(prepared_dir)
    selected = _selected(manifest, questions)
    train_keys, validation_keys = _v7_all_keys(selected, swe_root)
    from .source_utils import run_base_path

    overlay_path = Path(manifest["cost_overlay"])
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    if overlay.get("metadata", {}).get("memory_benchmark_version") != "128k_v2_tem0":
        raise ValueError("frozen cost overlay uses a different memory benchmark version")
    swe = load_examples(prepared_dir, swe_root, k, explorer=explorer)
    result = {split: [{**row, "bench": "swe_bench_pro", "probe_kind": "agent_prefix"}
                      for row in rows] for split, rows in swe.items()}
    errors = []
    for pid, item in selected.items():
        bench = item["bench"]
        if bench == "swe_bench_pro":
            continue
        if bench not in NON_SWE:
            raise ValueError(f"unknown selected benchmark: {bench}")
        source_split = "test" if pid >= 1100 else "train"
        key = item["prompt_key"]
        split = "test" if source_split == "test" else (
            "train" if key in train_keys else "validation")
        if source_split == "train" and key not in train_keys | validation_keys:
            raise ValueError(f"selected prompt absent from v7 source pool: {key}")
        group = records[pid]
        if any(group[mid]["category"] != item["category"] for mid in range(5)):
            raise ValueError(f"prepared category differs from selected category: {key}")
        iid = item["instance_id"]
        run = run_base_path(swe_root / "sweagent_results", explorer, bench,
                            f"{source_split}_split", "128k_v2_tem0")
        base = run / iid
        try:
            probe = read_complete_probe(base, iid, bench)
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{key}: {exc}")
            continue
        probe_cost = group[MODEL_ORDER.index(explorer)]["cost"]
        overlay_cost = overlay.get("splits", {}).get(source_split, {}).get(explorer, {}).get(key)
        if (not isinstance(overlay_cost, (int, float)) or isinstance(overlay_cost, bool)
                or not math.isclose(probe_cost, overlay_cost, rel_tol=0, abs_tol=1e-9)):
            raise ValueError(f"explorer frozen cost differs from pinned overlay: {key}")
        result[split].append({
            "prompt_id": pid, "instance_id": iid, "prompt_key": key,
            "bench": bench, "problem": item["prompt"],
            "labels": [group[mid]["label"] for mid in range(5)],
            "full_costs": [group[mid]["cost"] for mid in range(5)],
            "prefix_cost": probe_cost, "prefix_cost_status": "exact",
            "prefix_cost_source": "frozen_full_run",
            "prefix_cost_reason": "", "early_finish": False,
            "prefix_steps": [], "split": split, "explorer_model": explorer, **probe,
        })
    if errors:
        raise ValueError(f"{len(errors)} Flash probes unavailable or invalid:\n"
                         + "\n".join(errors))
    for split, rows in result.items():
        rows.sort(key=lambda row: row["prompt_id"])
    if {split: len(rows) for split, rows in result.items()} != FOUR_BENCH_COUNTS:
        raise ValueError("four-bench split counts differ from frozen 867/233/382 protocol")
    if Counter(row["bench"] for row in result["test"]) != TEST_BENCH_COUNTS:
        raise ValueError("four-bench test coverage differs from frozen comparison")
    if len({row["prompt_key"] for rows in result.values() for row in rows}) != 1482:
        raise ValueError("four-bench prompt keys are missing or duplicated")
    return result
