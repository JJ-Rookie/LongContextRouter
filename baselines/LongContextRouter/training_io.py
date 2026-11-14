"""Atomic checkpoint and metrics persistence for LongContextRouter training."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


RUN_SIGNATURE_FIELDS = (
    "models",
    "benches",
    "exclude_training_benches",
    "train_data_path",
    "test_data_path",
    "source_split",
    "run_dir",
    "limit_prompts_per_bench",
    "fill_missing_zero",
    "swe_bench_positive_tail_steps",
    "swe_bench_step_supervision_mode",
    "ifbench_mode",
    "memory_label",
    "memory_benchmark_version",
    "memory_f1_threshold",
    "cost_source",
    "cost_overrides_path",
    "cost_overrides_sha256",
    "require_cost",
    "fusion_mode",
    "batch_size",
    "num_workers",
    "split_ratio",
    "seed",
    "split_seed",
    "eval_every",
    "train_eval_every",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "label_threshold",
    "cost_loss_weight",
    "cost_ranking_margin",
    "cost_loss_warmup_epochs",
    "routing_group_sampler_mode",
    "max_routing_group_views",
    "save_best_metric",
    "embedder_model_name",
    "embedder_backend",
    "embedder_max_chunk_length_k",
    "embedder_overlap_tokens_k",
    "embedder_aggregator_n_layers",
    "embedder_aggregator_dropout",
    "embedder_chunk_batch_size",
    "embedder_chunk_workers",
    "no_embedder_qk_norm",
    "embedder_rope_base",
    "alpha",
)

RUN_SIGNATURE_DEFAULTS = {
    "exclude_training_benches": [],
    "cost_overrides_path": None,
    "cost_overrides_sha256": None,
    "memory_benchmark_version": "legacy",
    "cost_loss_weight": 0.0,
    "cost_ranking_margin": 0.1,
    "cost_loss_warmup_epochs": 0,
    # A missing field identifies checkpoints created before sampler behavior
    # became independently configurable.  It must not silently resume under a
    # new warmup/sampler schedule.
    "routing_group_sampler_mode": "legacy",
    "max_routing_group_views": 64,
    "save_best_metric": "sample_accuracy",
    "embedder_aggregator_dropout": 0.0,
    "swe_bench_positive_tail_steps": 0,
    "swe_bench_step_supervision_mode": None,
    "split_seed": None,
}


def cost_overrides_fingerprint(path: str | Path | None) -> str | None:
    """Identify the exact immutable billing overlay, when one is supplied."""
    return None if path is None else hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_fingerprint(datasets: Sequence[Any]) -> str:
    """Hash the ordered train/validation split and all training semantics."""
    digest = hashlib.sha256()
    for split_index, dataset in enumerate(datasets):
        digest.update(f"split:{split_index}:size:{len(dataset.items)}\n".encode("ascii"))
        for item in dataset.items:
            payload = {
                "prompt": item.prompt,
                "prompt_id": item.prompt_id,
                "category": item.category,
                "labels": item.labels.detach().cpu().tolist(),
                "costs": item.costs.detach().cpu().tolist(),
            }
            supervision_mask = getattr(item, "supervision_mask", None)
            routing_mask = getattr(item, "routing_mask", None)
            cost_available_mask = getattr(item, "cost_available_mask", None)
            routing_group_id = getattr(item, "routing_group_id", "")
            loss_targets = getattr(item, "loss_targets", None)
            loss_weights = getattr(item, "loss_weights", None)
            loss_group_id = getattr(item, "loss_group_id", "")
            tail_offset = getattr(item, "tail_offset", None)
            has_sparse_supervision = (
                supervision_mask is not None
                and not bool(supervision_mask.detach().cpu().all())
            )
            has_sparse_routing = (
                routing_mask is not None
                and not bool(routing_mask.detach().cpu().all())
            )
            has_missing_cost = (
                cost_available_mask is not None
                and not bool(cost_available_mask.detach().cpu().all())
            )
            has_custom_group = bool(
                routing_group_id and routing_group_id != item.prompt_id
            )
            if has_sparse_supervision or has_sparse_routing or has_custom_group:
                payload.update(
                    {
                        "supervision_mask": (
                            supervision_mask.detach().cpu().tolist()
                            if supervision_mask is not None
                            else None
                        ),
                        "routing_mask": (
                            routing_mask.detach().cpu().tolist()
                            if routing_mask is not None
                            else None
                        ),
                        "routing_group_id": routing_group_id,
                    }
                )
            if has_missing_cost:
                payload["cost_available_mask"] = (
                    cost_available_mask.detach().cpu().tolist()
                )
            has_soft_targets = (
                loss_targets is not None
                and not bool(
                    torch.equal(
                        loss_targets.detach().cpu(),
                        item.labels.detach().cpu().to(dtype=loss_targets.dtype),
                    )
                )
            )
            has_custom_loss_weights = (
                loss_weights is not None
                and supervision_mask is not None
                and not bool(
                    torch.equal(
                        loss_weights.detach().cpu()[
                            supervision_mask.detach().cpu()
                        ],
                        torch.ones_like(
                            loss_weights.detach().cpu()[
                                supervision_mask.detach().cpu()
                            ]
                        ),
                    )
                )
            )
            has_step_semantics = (
                has_soft_targets
                or has_custom_loss_weights
                or tail_offset is not None
                or bool(loss_group_id and loss_group_id != routing_group_id)
            )
            if has_step_semantics:
                payload.update(
                    {
                        "loss_targets": (
                            loss_targets.detach().cpu().tolist()
                            if loss_targets is not None
                            else None
                        ),
                        "loss_weights": (
                            loss_weights.detach().cpu().tolist()
                            if loss_weights is not None
                            else None
                        ),
                        "loss_group_id": loss_group_id,
                        "tail_offset": tail_offset,
                    }
                )
            digest.update(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def build_run_signature(
    args: Any,
    cache_fingerprint: str | None,
    data_fingerprint: str | None = None,
) -> dict[str, Any]:
    return {
        **{
            field: json_safe(
                getattr(args, field, RUN_SIGNATURE_DEFAULTS.get(field))
            )
            for field in RUN_SIGNATURE_FIELDS
        },
        "cache_fingerprint": cache_fingerprint,
        "data_fingerprint": data_fingerprint,
    }


def _normalize_run_signature(signature: Any) -> Any:
    if not isinstance(signature, Mapping):
        return signature
    normalized = dict(signature)
    for field, default in RUN_SIGNATURE_DEFAULTS.items():
        if normalized.get(field) is None:
            normalized[field] = default
    return normalized


def _run_signatures_match(actual: Any, expected: Mapping[str, Any]) -> bool:
    normalized_actual = _normalize_run_signature(actual)
    normalized_expected = _normalize_run_signature(expected)
    if normalized_actual == normalized_expected:
        return True
    if isinstance(normalized_actual, Mapping) and "data_fingerprint" not in normalized_actual:
        legacy_expected = dict(normalized_expected)
        legacy_expected.pop("data_fingerprint", None)
        return dict(normalized_actual) == legacy_expected
    return False


def save_training_checkpoint(
    *,
    path: str | Path,
    router: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: Any,
    completed_epoch: int,
    metrics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    best_metric: float,
    cache_fingerprint: str | None,
    train_generator: torch.Generator,
    data_fingerprint: str | None = None,
    best_cost_metric: float = float("inf"),
    checkpoint_selection_state: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "checkpoint_format_version": 1,
        "model_state_dict": router.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_epoch": int(completed_epoch),
        "metrics": json_safe(dict(metrics)),
        "history": json_safe(list(history)),
        "best_metric": float(best_metric),
        "best_cost_metric": float(best_cost_metric),
        "checkpoint_selection_state": json_safe(
            dict(checkpoint_selection_state or {})
        ),
        "args": json_safe(vars(args)),
        "run_signature": build_run_signature(
            args,
            cache_fingerprint,
            data_fingerprint,
        ),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "train_generator_state": train_generator.get_state(),
    }
    atomic_torch_save(payload, path)


def restore_training_checkpoint(
    *,
    path: str | Path,
    router: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: Any,
    cache_fingerprint: str | None,
    train_generator: torch.Generator,
    device: torch.device,
    data_fingerprint: str | None = None,
    return_best_cost_metric: bool = False,
    return_checkpoint_selection_state: bool = False,
) -> (
    tuple[int, list[dict[str, Any]], float]
    | tuple[int, list[dict[str, Any]], float, float]
    | tuple[int, list[dict[str, Any]], float, float, dict[str, Any]]
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_format_version") != 1:
        raise ValueError("unsupported training checkpoint format")
    expected_signature = build_run_signature(
        args,
        cache_fingerprint,
        data_fingerprint,
    )
    if not _run_signatures_match(checkpoint.get("run_signature"), expected_signature):
        raise ValueError("training checkpoint signature does not match this run")

    router.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

    random.setstate(checkpoint["python_random_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_states = checkpoint.get("cuda_rng_states", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    train_generator.set_state(checkpoint["train_generator_state"])

    completed_epoch = int(checkpoint["completed_epoch"])
    history = [dict(item) for item in checkpoint.get("history", [])]
    best_metric = float(checkpoint.get("best_metric", float("-inf")))
    best_cost_metric = float(checkpoint.get("best_cost_metric", float("inf")))
    checkpoint_selection_state = dict(
        checkpoint.get("checkpoint_selection_state") or {}
    )
    if return_checkpoint_selection_state:
        return (
            completed_epoch + 1,
            history,
            best_metric,
            best_cost_metric,
            checkpoint_selection_state,
        )
    if return_best_cost_metric:
        return completed_epoch + 1, history, best_metric, best_cost_metric
    return completed_epoch + 1, history, best_metric


def save_checkpoint(
    path: str | Path,
    router: torch.nn.Module,
    args: Any,
    epoch: int,
    metrics: Mapping[str, Any],
) -> None:
    atomic_torch_save(
        {
            "model_state_dict": router.state_dict(),
            "epoch": epoch,
            "metrics": json_safe(dict(metrics)),
            "args": json_safe(vars(args)),
        },
        path,
    )


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def default_best_model_path(model_save_path: str | Path) -> Path:
    path = Path(model_save_path)
    return path.with_name(f"{path.stem}_best{path.suffix or '.pth'}")


def default_best_accuracy_model_path(model_save_path: str | Path) -> Path:
    path = Path(model_save_path)
    return path.with_name(
        f"{path.stem}_best_accuracy{path.suffix or '.pth'}"
    )


def default_best_cost_aware_model_path(model_save_path: str | Path) -> Path:
    path = Path(model_save_path)
    return path.with_name(
        f"{path.stem}_best_cost_aware{path.suffix or '.pth'}"
    )


def default_last_model_path(model_save_path: str | Path) -> Path:
    path = Path(model_save_path)
    return path.with_name(f"{path.stem}_last{path.suffix or '.pth'}")
