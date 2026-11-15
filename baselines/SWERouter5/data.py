from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from baselines.SWERouter5.cost import (
    PrefixCostResult, actionable_indices, prefix_cost_from_artifacts,
)


MODEL_ORDER = (
    "deepseek-v4-pro", "deepseek-v4-flash-0731", "deepseek-v4-pro-0813",
    "qwen3.5-122b-a10b", "glm-4-flash-250414",
)
PREPARED_COUNTS = {"train": 867, "validation": 233, "test": 382}
SWE_COUNTS = {"train": 160, "validation": 40, "test": 100}
DATA_FIELDS = ("prompt_id", "model_id", "category_id", "label", "model_name",
               "category", "prompt", "cost")
SELECTED_FIELDS = ("prompt_id", "prompt_key", "bench", "instance_id", "category", "prompt")


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError(f"{path}: CSV schema differs from the frozen export")
        return list(reader)


def actionable_prefix(trajectory: list[dict], k: int) -> tuple[str, tuple[int, ...], bool]:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    indices = actionable_indices(trajectory)
    selected = indices[:k]
    blocks = []
    for number, index in enumerate(selected, 1):
        step = trajectory[index]
        fields = []
        for name in ("thought", "action", "observation"):
            value = step.get(name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"step {index} {name} is not text")
            fields.append(f"{name.capitalize()}: {value or ''}")
        blocks.append(f"Step {number}\n" + "\n".join(fields))
    return "\n\n".join(blocks), selected, len(indices) < k


MANIFEST_PATH_FIELDS = ("official_train_export", "official_test_export",
                        "reference_rows", "cost_overlay")


def resolve_manifest_paths(manifest: dict, prepared_dir: Path) -> dict:
    resolved = {}
    for field in MANIFEST_PATH_FIELDS:
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prepared manifest requires a nonempty {field} path")
        path = Path(value)
        resolved[field] = str(path if path.is_absolute() else Path(prepared_dir).resolve() / path)
    return {**manifest, **resolved}


def _prepared(prepared_dir: Path) -> tuple[dict, dict[int, dict[int, dict]], dict[int, str]]:
    manifest_path = prepared_dir / "manifest.json"
    manifest = resolve_manifest_paths(
        json.loads(manifest_path.read_text(encoding="utf-8")), prepared_dir)
    if manifest.get("model_order") != list(MODEL_ORDER):
        raise ValueError("prepared model order differs from frozen five-model pool")
    if manifest.get("prompt_counts") != PREPARED_COUNTS:
        raise ValueError("prepared split counts differ from old four-bench export")
    questions = _read_csv(prepared_dir / "question_order_ours.csv", ("prompt_id", "prompt"))
    if len(questions) != sum(PREPARED_COUNTS.values()) or any(
        row["prompt_id"] != str(index) or not row["prompt"]
        for index, row in enumerate(questions)
    ):
        raise ValueError("prepared question order is incomplete or reordered")
    question_text = {index: row["prompt"] for index, row in enumerate(questions)}
    by_id: dict[int, dict[int, dict]] = defaultdict(dict)
    seen = set()
    for split, expected in PREPARED_COUNTS.items():
        rows = _read_csv(prepared_dir / f"{split}_ours.csv", DATA_FIELDS)
        ids = set()
        for row in rows:
            pid, mid = int(row["prompt_id"]), int(row["model_id"])
            if not 0 <= pid < len(questions) or not 0 <= mid < 5 or mid in by_id[pid]:
                raise ValueError("prepared model matrix contains invalid or duplicate IDs")
            if row["model_name"] != MODEL_ORDER[mid] or row["prompt"] != question_text[pid]:
                raise ValueError("prepared model or question text differs from official order")
            label, cost = int(row["label"]), float(row["cost"])
            if label not in (0, 1) or not math.isfinite(cost) or cost < 0:
                raise ValueError("invalid frozen label or cost")
            if row["category_id"] != row["category"]:
                raise ValueError("category ID differs from category")
            by_id[pid][mid] = {**row, "label": label, "cost": cost}
            ids.add(pid)
        if len(ids) != expected or seen & ids:
            raise ValueError("prepared prompt split coverage differs from manifest")
        seen.update(ids)
    if seen != set(question_text) or any(set(group) != set(range(5)) for group in by_id.values()):
        raise ValueError("prepared five-model matrix is incomplete")
    return manifest, by_id, question_text


def _selected(manifest: dict, question_text: dict[int, str]) -> dict[int, dict]:
    selected = {}
    offset = PREPARED_COUNTS["train"] + PREPARED_COUNTS["validation"]
    for source, shift, count in (("train", 0, offset), ("test", offset, PREPARED_COUNTS["test"])):
        path = Path(manifest[f"official_{source}_export"]) / "selected_prompts.csv"
        rows = _read_csv(path, SELECTED_FIELDS)
        if len(rows) != count:
            raise ValueError("official selected prompt count differs from frozen export")
        for index, row in enumerate(rows):
            pid = shift + index
            if row["prompt_id"] != str(index) or row["prompt"] != question_text[pid]:
                raise ValueError("official selected ID or prompt differs from prepared CSV")
            if row["prompt_key"] != f"{row['bench']}:{row['instance_id']}":
                raise ValueError("official selected prompt key is inconsistent")
            selected[pid] = row
    keys = [row["prompt_key"] for row in selected.values()]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate official selected prompt key")
    return selected


def _v7_split(selected: dict[int, dict], swe_root: Path) -> tuple[set[str], set[str]]:
    from .source_utils import split_prompt_keys
    from types import SimpleNamespace

    categories = {row["prompt_key"]: row["category"] for pid, row in selected.items()
                  if pid < PREPARED_COUNTS["train"] + PREPARED_COUNTS["validation"]}
    ifbench = swe_root / "data/benchmark_splits/ifbench/train.jsonl"
    with ifbench.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            iid = str(row.get("key", row.get("id", "")))
            labels = row.get("instruction_id_list") or []
            tier = str(labels[0]).split(":", 1)[0] if labels else "unknown"
            key = f"ifbench:{iid}"
            if not iid or key in categories:
                raise ValueError("invalid or duplicate IFBench key")
            categories[key] = f"ifbench/{tier}"
    counts = Counter(key.split(":", 1)[0] for key in categories)
    if counts != {"bbh": 500, "ifbench": 200, "memory_agent_bench": 200,
                  "mmau": 200, "swe_bench_pro": 200}:
        raise ValueError(f"v7 source prompt pool differs from expected 1300 keys: {counts}")
    items = {key: SimpleNamespace(category=value) for key, value in categories.items()}
    train, validation = split_prompt_keys(sorted(categories), items, 0.8, 43)
    swe_train = {key for key in train if key.startswith("swe_bench_pro:")}
    swe_validation = {key for key in validation if key.startswith("swe_bench_pro:")}
    if len(swe_train) != 160 or len(swe_validation) != 40:
        raise ValueError("v7 SWE split has unexpected size")
    return swe_train, swe_validation


def _test_reference(manifest: dict, selected: dict[int, dict], records: dict[int, dict[int, dict]]) -> None:
    path = Path(manifest["reference_rows"])
    pool = json.loads(path.read_text(encoding="utf-8"))["pool"]
    for mid, model in enumerate(MODEL_ORDER):
        rows = [records[pid][mid] for pid, item in selected.items()
                if pid >= 1100 and item["bench"] == "swe_bench_pro"]
        expected = pool[model]["swe_bench_pro"]
        if len(rows) != expected[0] or sum(row["label"] for row in rows) != expected[1] or not math.isclose(
            sum(row["cost"] for row in rows), expected[2], rel_tol=0, abs_tol=1e-7
        ):
            raise ValueError(f"{model}: SWE test rows differ from old five-model reference")


def load_examples(
    prepared_dir: Path, swe_root: Path, k: int,
    explorer: str = "deepseek-v4-flash-0731", *, load_prefixes: bool = True,
) -> dict[str, list[dict]]:
    prepared_dir, swe_root = Path(prepared_dir), Path(swe_root)
    if explorer not in ("deepseek-v4-flash-0731", "deepseek-v4-pro-0813"):
        raise ValueError("unsupported SWE explorer model")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    manifest, records, questions = _prepared(prepared_dir)
    selected = _selected(manifest, questions)
    train_keys, validation_keys = _v7_split(selected, swe_root)
    _test_reference(manifest, selected, records)
    overlay = Path(manifest["cost_overlay"])
    rates = json.loads(overlay.read_text(encoding="utf-8"))["metadata"]["prices"][explorer]
    result = {"train": [], "validation": [], "test": []}
    for pid, item in selected.items():
        if item["bench"] != "swe_bench_pro":
            continue
        source_split = "test" if pid >= 1100 else "train"
        split = "test" if source_split == "test" else (
            "train" if item["prompt_key"] in train_keys else "validation")
        if split == "validation" and item["prompt_key"] not in validation_keys:
            raise ValueError("SWE prompt missing from v7 split")
        group = records[pid]
        if any(group[mid]["category"] != item["category"] for mid in range(5)):
            raise ValueError("prepared category differs from official selected category")
        base = (swe_root / "sweagent_results" / explorer / "swe_bench_pro" /
                f"{source_split}_split" / item["instance_id"])
        traj, trace = base / f"{item['instance_id']}.traj", base / f"{item['instance_id']}.trace.log"
        prefix, early, billing, prefix_steps = "", False, None, []
        if load_prefixes:
            try:
                data = json.loads(traj.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("info"), dict):
                    raise ValueError("trajectory root or info is malformed")
                if data["info"].get("instance_id") != item["instance_id"]:
                    raise ValueError("trajectory instance ID differs from selected prompt")
                prefix, indices, early = actionable_prefix(data["trajectory"], k)
                prefix_steps = [
                    {name: data["trajectory"][index].get(name) or ""
                     for name in ("thought", "action", "observation")}
                    for index in indices
                ]
                billing = prefix_cost_from_artifacts(
                    traj, trace, k, group[MODEL_ORDER.index(explorer)]["cost"], rates)
            except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
                billing = PrefixCostResult(None, "unavailable", f"trajectory unavailable: {exc}")
        result[split].append({
            "prompt_id": pid, "instance_id": item["instance_id"],
            "prompt_key": item["prompt_key"], "problem": item["prompt"],
            "prefix_text": prefix, "labels": [group[mid]["label"] for mid in range(5)],
            "prefix_steps": prefix_steps,
            "full_costs": [group[mid]["cost"] for mid in range(5)],
            "prefix_cost": billing.cost if billing else None,
            "prefix_cost_status": billing.status if billing else "not_loaded",
            "prefix_cost_reason": billing.reason if billing else "",
            "explorer_model": explorer,
            "early_finish": early, "split": split,
        })
    if {key: len(value) for key, value in result.items()} != SWE_COUNTS:
        raise ValueError("SWE example counts differ from v7 160/40/100 protocol")
    return result
