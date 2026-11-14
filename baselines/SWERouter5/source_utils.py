"""Minimal source-layout and split helpers retained from the data exporter."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

MEMORY_BENCHMARK_VERSIONS = ("legacy", "128k_v2", "128k_v2_tem0")
MEMORY_TEM0_RESULT_MODELS = frozenset({
    "deepseek-v4-pro", "deepseek-v4-flash-0731", "deepseek-v4-pro-0813",
})


def validate_memory_benchmark_version(version: str) -> None:
    if version not in MEMORY_BENCHMARK_VERSIONS:
        raise ValueError(f"unsupported MemoryAgentBench version: {version!r}")


def run_base_path(
    results_root: Path, model: str, bench: str, run_dir: str,
    memory_benchmark_version: str = "legacy",
) -> Path:
    validate_memory_benchmark_version(memory_benchmark_version)
    result_bench = bench
    if bench == "memory_agent_bench":
        result_bench = {
            "128k_v2": "memory_agent_bench_128k_v2",
            "128k_v2_tem0": (
                "memory_agent_bench_128k_v2_tem0"
                if model in MEMORY_TEM0_RESULT_MODELS
                else "memory_agent_bench_128k_v2"
            ),
        }.get(memory_benchmark_version, bench)
    return results_root / model / result_bench / run_dir


def split_prompt_keys(
    prompt_keys: list[str],
    prompt_items: dict[str, Any],
    train_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    by_category: dict[str, list[str]] = defaultdict(list)
    for key in prompt_keys:
        by_category[prompt_items[key].category].append(key)

    train_keys: set[str] = set()
    test_keys: set[str] = set()
    for _category, keys in sorted(by_category.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n_train = int(len(keys) * train_ratio)
        if len(keys) > 1 and n_train == len(keys):
            n_train -= 1
        train_keys.update(keys[:n_train])
        test_keys.update(keys[n_train:])
    return train_keys, test_keys
