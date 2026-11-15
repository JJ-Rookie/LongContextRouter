from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from baselines.LongContextRouter.training_io import (
    default_best_accuracy_model_path,
    default_best_cost_aware_model_path,
    default_best_model_path,
    default_last_model_path,
    json_safe as _json_safe,
    restore_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
    write_json,
)
from baselines.LongContextRouter.op_losses import (
    LambdaSampler,
    decision_aligned_loss,
    listwise_selection_loss,
)


TASK_AUX_LABEL_MAP = {
    "bbh": 0,
    "memory_agent_bench": 1,
    "mmau": 2,
    "swe_bench_pro": 3,
}


DEFAULT_SWE_AGENT_ROOT = Path("data/source")
SUPPORTED_BENCHES = ("bbh", "ifbench", "memory_agent_bench", "mmau", "swe_bench_pro")
DEFAULT_DIRECT_BENCHES = ("bbh", "ifbench", "memory_agent_bench", "mmau")
LONG_CONTEXT_MEMORY_PROMPT_MODE = "question_context"
SWE_BENCH_STEP_SUPERVISION_MODES = ("single_hard", "all_steps_soft")
SWE_BENCH_TRAINING_PROMPT_MODES = ("legacy", "problem_only")


def configure_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


@dataclass(frozen=True)
class PromptBatchItem:
    prompt: str
    labels: torch.Tensor
    category: str
    costs: torch.Tensor
    prompt_id: str = ""
    supervision_mask: torch.Tensor | None = None
    routing_mask: torch.Tensor | None = None
    cost_available_mask: torch.Tensor | None = None
    routing_group_id: str = ""
    loss_targets: torch.Tensor | None = None
    loss_weights: torch.Tensor | None = None
    loss_group_id: str = ""
    tail_offset: int | None = None


@dataclass(frozen=True)
class PromptBatch:

    prompts: list[str]
    labels: torch.Tensor
    categories: list[str]
    costs: torch.Tensor
    supervision_mask: torch.Tensor
    routing_mask: torch.Tensor
    cost_available_mask: torch.Tensor
    routing_group_ids: list[str]
    loss_targets: torch.Tensor
    loss_weights: torch.Tensor
    loss_group_ids: list[str]
    tail_offsets: list[int | None]

    def __iter__(self):
        return iter((self.prompts, self.labels, self.categories, self.costs))

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return (self.prompts, self.labels, self.categories, self.costs)[index]


class LongContextRouterDataset(Dataset):

    REQUIRED_COLUMNS = {
        "prompt_id",
        "model_id",
        "label",
        "model_name",
        "category",
        "prompt",
    }

    def __init__(
        self,
        csv_path: str | Path,
        num_models: int | None = None,
        label_threshold: float = 0.5,
    ):
        self.csv_path = Path(csv_path)
        self.label_threshold = label_threshold
        rows = self._read_rows(self.csv_path)
        self.num_models = num_models or infer_num_models([self.csv_path])
        if self.num_models <= 0:
            raise ValueError("num_models must be positive")

        self.model_names: dict[int, str] = {}
        self.items = self._build_items(rows)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Mapping[str, str]],
        num_models: int | None = None,
        label_threshold: float = 0.5,
        source_name: str = "in-memory rows",
    ) -> "LongContextRouterDataset":
        if not rows:
            raise ValueError(f"{source_name} contains no training rows")
        missing = cls.REQUIRED_COLUMNS - set(rows[0])
        if missing:
            raise ValueError(f"{source_name} missing required columns: {sorted(missing)}")

        dataset = cls.__new__(cls)
        dataset.csv_path = Path(source_name)
        dataset.label_threshold = label_threshold
        dataset.num_models = num_models or infer_num_models_from_rows(rows)
        if dataset.num_models <= 0:
            raise ValueError("num_models must be positive")
        dataset.model_names = {}
        dataset.items = dataset._build_items(rows)
        return dataset

    def _read_rows(self, csv_path: Path) -> list[dict[str, str]]:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        configure_csv_field_size_limit()
        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"{csv_path} contains no training rows")
        return rows

    def _build_items(self, rows: Sequence[Mapping[str, str]]) -> list[PromptBatchItem]:
        grouped: dict[str, list[Mapping[str, str]]] = {}
        for row in rows:
            prompt_id = str(row["prompt_id"])
            grouped.setdefault(prompt_id, []).append(row)

        items: list[PromptBatchItem] = []
        for prompt_id, group in grouped.items():
            labels = torch.zeros(self.num_models, dtype=torch.long)
            loss_targets = torch.zeros(self.num_models, dtype=torch.float32)
            loss_weights = torch.zeros(self.num_models, dtype=torch.float32)
            costs = torch.zeros(self.num_models, dtype=torch.float64)
            prompt = str(group[0]["prompt"])
            category = str(group[0].get("category") or group[0].get("category_id") or "unknown")
            model_specific = parse_bool_flag(
                group[0].get("model_specific"),
                default=False,
            )
            supervision_mask = torch.zeros(self.num_models, dtype=torch.bool)
            routing_mask = torch.zeros(self.num_models, dtype=torch.bool)
            cost_available_mask = torch.zeros(self.num_models, dtype=torch.bool)
            if not model_specific:
                supervision_mask.fill_(True)
                routing_mask.fill_(True)
                loss_weights.fill_(1.0)
            routing_group_id = str(
                group[0].get("routing_group_id") or prompt_id
            )
            loss_group_id = str(
                group[0].get("loss_group_id") or routing_group_id
            )
            raw_tail_offset = group[0].get("tail_offset")
            tail_offset = (
                int(raw_tail_offset)
                if raw_tail_offset is not None and str(raw_tail_offset).strip()
                else None
            )
            if tail_offset is not None and tail_offset <= 0:
                raise ValueError(
                    f"tail offset must be positive for prompt_id={prompt_id}"
                )
            seen_models: set[int] = set()

            for row in group:
                if str(row["prompt"]) != prompt:
                    raise ValueError(
                        f"inconsistent prompt text for prompt_id={prompt_id} in {self.csv_path}"
                    )
                row_category = str(
                    row.get("category") or row.get("category_id") or "unknown"
                )
                if row_category != category:
                    raise ValueError(
                        f"inconsistent category for prompt_id={prompt_id} in {self.csv_path}"
                    )
                row_model_specific = parse_bool_flag(
                    row.get("model_specific"),
                    default=False,
                )
                if row_model_specific != model_specific:
                    raise ValueError(
                        f"inconsistent model_specific for prompt_id={prompt_id} "
                        f"in {self.csv_path}"
                    )
                row_group_id = str(row.get("routing_group_id") or prompt_id)
                if row_group_id != routing_group_id:
                    raise ValueError(
                        f"inconsistent routing_group_id for prompt_id={prompt_id} "
                        f"in {self.csv_path}"
                    )
                row_loss_group_id = str(
                    row.get("loss_group_id") or row_group_id
                )
                if row_loss_group_id != loss_group_id:
                    raise ValueError(
                        f"inconsistent loss_group_id for prompt_id={prompt_id} "
                        f"in {self.csv_path}"
                    )
                row_raw_tail_offset = row.get("tail_offset")
                row_tail_offset = (
                    int(row_raw_tail_offset)
                    if row_raw_tail_offset is not None
                    and str(row_raw_tail_offset).strip()
                    else None
                )
                if row_tail_offset != tail_offset:
                    raise ValueError(
                        f"inconsistent tail_offset for prompt_id={prompt_id} "
                        f"in {self.csv_path}"
                    )
                model_id = parse_model_id(row["model_id"], self.num_models)
                if model_id in seen_models:
                    raise ValueError(
                        f"duplicate model_id={model_id} for prompt_id={prompt_id} in {self.csv_path}"
                    )
                seen_models.add(model_id)
                hard_label = int(
                    parse_float(row.get("label", "0")) >= self.label_threshold
                )
                labels[model_id] = hard_label
                raw_loss_target = row.get("loss_target")
                loss_target = (
                    float(hard_label)
                    if raw_loss_target is None or not str(raw_loss_target).strip()
                    else parse_float(raw_loss_target)
                )
                if not math.isfinite(loss_target) or not 0.0 <= loss_target <= 1.0:
                    raise ValueError(
                        f"loss target must be finite and in [0, 1] for "
                        f"prompt_id={prompt_id}, model_id={model_id}"
                    )
                raw_loss_weight = row.get("loss_weight")
                loss_weight = (
                    1.0
                    if raw_loss_weight is None or not str(raw_loss_weight).strip()
                    else parse_float(raw_loss_weight)
                )
                if not math.isfinite(loss_weight) or loss_weight <= 0:
                    raise ValueError(
                        f"loss weight must be positive and finite for "
                        f"prompt_id={prompt_id}, model_id={model_id}"
                    )
                loss_targets[model_id] = loss_target
                loss_weights[model_id] = loss_weight
                raw_cost = row.get("cost")
                if raw_cost is not None and str(raw_cost).strip():
                    cost = parse_float(raw_cost)
                    if not math.isfinite(cost) or cost < 0:
                        raise ValueError(
                            f"invalid cost={cost!r} for prompt_id={prompt_id}, "
                            f"model_id={model_id} in {self.csv_path}"
                        )
                    costs[model_id] = cost
                    cost_available_mask[model_id] = True
                self.model_names.setdefault(model_id, str(row.get("model_name") or model_id))
                if model_specific:
                    supervision_mask[model_id] = True
                    if parse_bool_flag(row.get("is_canonical"), default=True):
                        routing_mask[model_id] = True

            items.append(
                PromptBatchItem(
                    prompt=prompt,
                    labels=labels,
                    category=category,
                    costs=costs,
                    prompt_id=prompt_id,
                    supervision_mask=supervision_mask,
                    routing_mask=routing_mask,
                    cost_available_mask=cost_available_mask,
                    routing_group_id=routing_group_id,
                    loss_targets=loss_targets,
                    loss_weights=loss_weights,
                    loss_group_id=loss_group_id,
                    tail_offset=tail_offset,
                )
            )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> PromptBatchItem:
        return self.items[index]


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def parse_bool_flag(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean flag: {value!r}")


def parse_model_id(value: Any, num_models: int) -> int:
    model_id = int(value)
    if model_id < 0 or model_id >= num_models:
        raise ValueError(f"model_id={model_id} is outside [0, {num_models})")
    return model_id


def infer_num_models_from_rows(rows: Iterable[Mapping[str, Any]]) -> int:
    max_model_id = -1
    for row in rows:
        max_model_id = max(max_model_id, int(row["model_id"]))
    if max_model_id < 0:
        raise ValueError("could not infer model count from rows")
    return max_model_id + 1


def infer_num_models(csv_paths: Iterable[str | Path]) -> int:
    max_model_id = -1
    configure_csv_field_size_limit()
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "model_id" not in (reader.fieldnames or []):
                raise ValueError(f"{path} missing required column: model_id")
            for row in reader:
                max_model_id = max(max_model_id, int(row["model_id"]))
    if max_model_id < 0:
        raise ValueError("could not infer model count from CSV files")
    return max_model_id + 1


def exporter_records_to_rows(
    records: Iterable[Any],
    prompt_items: Mapping[str, Any],
    model_to_id: Mapping[str, int],
    selected_prompt_keys: set[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        if record.prompt_key not in selected_prompt_keys:
            continue
        item = prompt_items[record.prompt_key]
        category = str(item.category)
        rows.append(
            {
                "prompt_id": str(record.prompt_key),
                "model_id": str(model_to_id[record.model_name]),
                "category_id": category,
                "label": str(int(record.label)),
                "model_name": str(record.model_name),
                "category": category,
                "prompt": str(item.prompt),
                "cost": str(float(getattr(record, "cost", 0.0))),
            }
        )
    return rows


def expanded_embedding_records_to_rows(
    records: Iterable[Any],
    split: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        if record.split != split:
            continue
        category = str(record.category)
        rows.append(
            {
                "prompt_id": f"expanded:{record.embedding_id}",
                "model_id": str(record.model_id),
                "category_id": category,
                "label": str(int(record.label)),
                "model_name": str(record.model_name),
                "category": category,
                "prompt": str(record.prompt),
                "cost": str(float(record.cost)),
                "model_specific": str(int(record.bench == "swe_bench_pro")),
                "routing_group_id": str(record.prompt_key),
                "is_canonical": str(int(record.is_canonical)),
                "loss_target": str(float(record.loss_target)),
                "loss_weight": str(float(record.loss_weight)),
                "loss_group_id": str(record.loss_group_id),
                "tail_offset": (
                    "" if record.tail_offset is None else str(record.tail_offset)
                ),
            }
        )
    return rows


def resolve_data_source(args: argparse.Namespace) -> str:
    has_csv_paths = bool(args.train_data_path or args.test_data_path)
    has_direct_source = bool(args.models)
    if has_csv_paths and has_direct_source:
        raise ValueError("provide either CSV paths or direct SWE-agent options, not both")
    if has_csv_paths:
        if getattr(args, "exclude_training_benches", ()):
            raise ValueError("--exclude-training-benches requires direct SWE-agent data, not CSV")
        if getattr(args, "cost_overrides_path", None) is not None:
            raise ValueError("--cost-overrides-path requires direct SWE-agent data, not CSV")
        if not args.train_data_path or not args.test_data_path:
            raise ValueError("--train-data-path and --test-data-path must be provided together")
        return "csv"
    if has_direct_source:
        return "direct_swe_agent"
    raise ValueError("provide either CSV paths or direct SWE-agent options with --models")


def build_csv_datasets(
    args: argparse.Namespace,
) -> tuple[LongContextRouterDataset, LongContextRouterDataset, int, dict[int, str]]:
    num_models = args.model_num or infer_num_models([args.train_data_path, args.test_data_path])
    train_dataset = LongContextRouterDataset(
        args.train_data_path,
        num_models=num_models,
        label_threshold=args.label_threshold,
    )
    test_dataset = LongContextRouterDataset(
        args.test_data_path,
        num_models=num_models,
        label_threshold=args.label_threshold,
    )
    model_names = {**train_dataset.model_names, **test_dataset.model_names}
    return train_dataset, test_dataset, num_models, model_names


def _validate_training_bench_exclusions(exclusions: Sequence[str]) -> frozenset[str]:
    if isinstance(exclusions, str) or not isinstance(exclusions, Sequence):
        raise ValueError("Training benchmark exclusions must be a sequence of benchmark names")
    if not all(isinstance(bench, str) for bench in exclusions):
        raise ValueError("Training benchmark exclusions must contain only benchmark names")
    excluded = frozenset(exclusions)
    if len(excluded) != len(exclusions):
        raise ValueError("Duplicate training benchmark exclusions are not allowed")
    if not excluded <= frozenset(SUPPORTED_BENCHES):
        raise ValueError(f"Unknown training benchmark exclusions: {sorted(excluded - set(SUPPORTED_BENCHES))}")
    return excluded


def exclude_training_bench_datasets(
    train_dataset: LongContextRouterDataset,
    validation_dataset: LongContextRouterDataset,
    excluded_benches: Sequence[str],
) -> tuple[LongContextRouterDataset, LongContextRouterDataset]:
    excluded = _validate_training_bench_exclusions(excluded_benches)
    if not excluded:
        return train_dataset, validation_dataset
    subsets = []
    for split, source in (("train", train_dataset), ("validation", validation_dataset)):
        remaining = [item for item in source.items if item.category.split("/", 1)[0] not in excluded]
        if not remaining:
            raise ValueError(f"Training benchmark exclusions leave the {split} split empty")
        subset = type(source).__new__(type(source))
        subset.__dict__ = {**vars(source), "items": remaining, "model_names": dict(source.model_names)}
        subsets.append(subset)
    return subsets[0], subsets[1]


def build_direct_swe_agent_datasets(args):
    raise ValueError("This release uses prepared CSV files. Use release train with --train-csv and --validation-csv.")


def build_datasets_from_args(
    args: argparse.Namespace,
) -> tuple[LongContextRouterDataset, LongContextRouterDataset, int, dict[int, str]]:
    data_source = resolve_data_source(args)
    if data_source == "csv":
        return build_csv_datasets(args)
    return build_direct_swe_agent_datasets(args)


def collate_prompt_batch(
    batch: Sequence[PromptBatchItem],
) -> PromptBatch:
    prompts = [item.prompt for item in batch]
    labels = torch.stack([item.labels for item in batch], dim=0)
    categories = [item.category for item in batch]
    costs = torch.stack([item.costs for item in batch], dim=0)
    supervision_mask = torch.stack(
        [
            item.supervision_mask
            if item.supervision_mask is not None
            else torch.ones_like(item.labels, dtype=torch.bool)
            for item in batch
        ],
        dim=0,
    )
    routing_mask = torch.stack(
        [
            item.routing_mask
            if item.routing_mask is not None
            else torch.ones_like(item.labels, dtype=torch.bool)
            for item in batch
        ],
        dim=0,
    )
    cost_available_mask = torch.stack(
        [
            item.cost_available_mask
            if item.cost_available_mask is not None
            else torch.ones_like(item.labels, dtype=torch.bool)
            for item in batch
        ],
        dim=0,
    )
    routing_group_ids = [
        item.routing_group_id or item.prompt_id or f"prompt:{item.prompt}"
        for item in batch
    ]
    loss_targets = torch.stack(
        [
            item.loss_targets
            if item.loss_targets is not None
            else item.labels.to(dtype=torch.float32)
            for item in batch
        ],
        dim=0,
    )
    loss_weights = torch.stack(
        [
            item.loss_weights
            if item.loss_weights is not None
            else torch.ones_like(item.labels, dtype=torch.float32)
            for item in batch
        ],
        dim=0,
    )
    loss_group_ids = [
        item.loss_group_id
        or item.routing_group_id
        or item.prompt_id
        or f"prompt:{item.prompt}"
        for item in batch
    ]
    tail_offsets = [item.tail_offset for item in batch]
    return PromptBatch(
        prompts=prompts,
        labels=labels,
        categories=categories,
        costs=costs,
        supervision_mask=supervision_mask,
        routing_mask=routing_mask,
        cost_available_mask=cost_available_mask,
        routing_group_ids=routing_group_ids,
        loss_targets=loss_targets,
        loss_weights=loss_weights,
        loss_group_ids=loss_group_ids,
        tail_offsets=tail_offsets,
    )


def select_models_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.size(-1) != 2:
        raise ValueError(f"expected logits with shape [batch, models, 2], got {tuple(logits.shape)}")
    success_margin = logits[:, :, 1] - logits[:, :, 0]
    return success_margin.argmax(dim=1)


def supervised_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    supervision_mask: torch.Tensor,
    *,
    loss_targets: torch.Tensor | None = None,
    loss_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 3 or logits.size(-1) != 2:
        raise ValueError(
            f"expected logits with shape [batch, models, 2], got {tuple(logits.shape)}"
        )
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} does not match logits "
            f"{tuple(logits.shape[:2])}"
        )
    if supervision_mask.shape != labels.shape:
        raise ValueError(
            f"supervision mask shape {tuple(supervision_mask.shape)} does not "
            f"match labels {tuple(labels.shape)}"
        )
    supervision_mask = supervision_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(supervision_mask.any()):
        raise ValueError("training batch contains no supervised model pairs")
    if loss_targets is None and loss_weights is None:
        return F.cross_entropy(logits[supervision_mask], labels[supervision_mask])

    targets = (
        labels.to(dtype=logits.dtype)
        if loss_targets is None
        else loss_targets.to(device=logits.device, dtype=logits.dtype)
    )
    weights = (
        torch.ones_like(targets)
        if loss_weights is None
        else loss_weights.to(device=logits.device, dtype=logits.dtype)
    )
    if targets.shape != labels.shape:
        raise ValueError(
            f"loss targets shape {tuple(targets.shape)} does not match labels "
            f"{tuple(labels.shape)}"
        )
    if weights.shape != labels.shape:
        raise ValueError(
            f"loss weights shape {tuple(weights.shape)} does not match labels "
            f"{tuple(labels.shape)}"
        )
    supervised_targets = targets[supervision_mask]
    supervised_weights = weights[supervision_mask]
    if not bool(torch.isfinite(supervised_targets).all()) or not bool(
        ((supervised_targets >= 0) & (supervised_targets <= 1)).all()
    ):
        raise ValueError("loss targets must be finite and in [0, 1]")
    if not bool(torch.isfinite(supervised_weights).all()) or not bool(
        (supervised_weights > 0).all()
    ):
        raise ValueError("loss weights must be positive and finite")
    success_margin = logits[:, :, 1] - logits[:, :, 0]
    terms = F.binary_cross_entropy_with_logits(
        success_margin[supervision_mask],
        supervised_targets,
        reduction="none",
    )
    return (terms * supervised_weights).sum() / supervised_weights.sum()


@dataclass(frozen=True)
class CostRankingLoss:
    loss: torch.Tensor
    eligible_group_count: int
    pair_count: int


@dataclass(frozen=True)
class TrainingLossMetrics:
    total_loss: float
    accuracy_loss: float
    cost_rank_loss: float
    supervised_pair_count: int
    cost_rank_group_count: int
    cost_rank_pair_count: int
    selection_loss: float = 0.0
    selection_view_count: int = 0
    decision_loss: float = 0.0
    decision_view_count: int = 0
    decision_lambda: float = 0.0
    task_aux_loss: float = 0.0
    task_aux_view_count: int = 0


def success_margins(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.size(-1) != 2:
        raise ValueError(
            f"expected logits with shape [batch, models, 2], got {tuple(logits.shape)}"
        )
    return logits[:, :, 1] - logits[:, :, 0]


def cost_pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    costs: torch.Tensor,
    supervision_mask: torch.Tensor,
    routing_mask: torch.Tensor,
    routing_group_ids: Sequence[str],
    cost_available_mask: torch.Tensor | None = None,
    margin: float = 0.1,
) -> CostRankingLoss:
    margins = success_margins(logits)
    expected_shape = logits.shape[:2]
    named_tensors = {
        "labels": labels,
        "costs": costs,
        "supervision mask": supervision_mask,
        "routing mask": routing_mask,
    }
    for name, tensor in named_tensors.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match logits "
                f"{tuple(expected_shape)}"
            )
    if len(routing_group_ids) != logits.size(0):
        raise ValueError(
            "routing group count does not match batch size: "
            f"{len(routing_group_ids)} != {logits.size(0)}"
        )
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("cost ranking margin must be non-negative and finite")

    supervision_mask = supervision_mask.to(device=logits.device, dtype=torch.bool)
    routing_mask = routing_mask.to(device=logits.device, dtype=torch.bool)
    labels = labels.to(device=logits.device)
    costs = costs.to(device=logits.device)
    if cost_available_mask is None:
        cost_available_mask = torch.ones_like(labels, dtype=torch.bool)
    elif cost_available_mask.shape != expected_shape:
        raise ValueError(
            f"cost available mask shape {tuple(cost_available_mask.shape)} does "
            f"not match logits {tuple(expected_shape)}"
        )
    cost_available_mask = cost_available_mask.to(
        device=logits.device,
        dtype=torch.bool,
    )

    candidates_by_group: dict[str, dict[int, tuple[torch.Tensor, int, float, bool, bool]]] = {}
    sparse_groups: set[str] = set()
    for row_index, group_id_value in enumerate(routing_group_ids):
        if not bool(routing_mask[row_index].any()):
            continue
        group_id = str(group_id_value)
        group_candidates = candidates_by_group.setdefault(group_id, {})
        if not bool(routing_mask[row_index].all()):
            sparse_groups.add(group_id)
        for model_id in torch.nonzero(
            routing_mask[row_index],
            as_tuple=False,
        ).flatten().tolist():
            if model_id in group_candidates:
                raise ValueError(
                    f"routing group {group_id!r} contains duplicate canonical "
                    f"candidate for model_id={model_id}"
                )
            group_candidates[model_id] = (
                margins[row_index, model_id],
                int(labels[row_index, model_id].item()),
                float(costs[row_index, model_id].item()),
                bool(supervision_mask[row_index, model_id]),
                bool(cost_available_mask[row_index, model_id]),
            )

    expected_model_ids = set(range(logits.size(1)))
    group_losses: list[torch.Tensor] = []
    total_pair_count = 0
    for group_id, candidates in candidates_by_group.items():
        if group_id in sparse_groups and set(candidates) != expected_model_ids:
            raise ValueError(
                f"cost-aware batch contains incomplete routing group {group_id!r}; "
                "use group-preserving batching"
            )

        correct_candidates: list[tuple[int, torch.Tensor, float]] = []
        for model_id, (score, label, cost, supervised, cost_available) in candidates.items():
            if not supervised or label != 1:
                continue
            if not cost_available:
                raise ValueError(
                    f"missing cost for correct candidate model_id={model_id} "
                    f"in routing group {group_id!r}"
                )
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(
                    f"invalid cost={cost!r} for model_id={model_id} "
                    f"in routing group {group_id!r}"
                )
            correct_candidates.append((model_id, score, cost))

        pair_losses: list[torch.Tensor] = []
        for left_index, (_left_id, left_score, left_cost) in enumerate(
            correct_candidates
        ):
            for _right_id, right_score, right_cost in correct_candidates[
                left_index + 1 :
            ]:
                if math.isclose(
                    left_cost,
                    right_cost,
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                ):
                    continue
                cheaper_score, costlier_score = (
                    (left_score, right_score)
                    if left_cost < right_cost
                    else (right_score, left_score)
                )
                pair_losses.append(
                    F.softplus(margin - (cheaper_score - costlier_score))
                )
        if pair_losses:
            group_losses.append(torch.stack(pair_losses).mean())
            total_pair_count += len(pair_losses)

    if not group_losses:
        return CostRankingLoss(
            loss=logits.sum() * 0.0,
            eligible_group_count=0,
            pair_count=0,
        )
    return CostRankingLoss(
        loss=torch.stack(group_losses).mean(),
        eligible_group_count=len(group_losses),
        pair_count=total_pair_count,
    )


def train_one_epoch(
    router: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model_ids: torch.Tensor,
    device: torch.device,
    gradient_clip_norm: float | None = None,
    cost_loss_weight: float = 0.0,
    cost_ranking_margin: float = 0.1,
    selection_loss_weight: float = 0.0,
    selection_temperature: float = 1.0,
    decision_loss_weight: float = 0.0,
    decision_lambda_sampler=None,
    task_aux_head: torch.nn.Module | None = None,
    task_aux_weight: float = 0.0,
    task_label_map: dict | None = None,
    return_metrics: bool = False,
) -> float | TrainingLossMetrics:
    if not math.isfinite(cost_loss_weight) or cost_loss_weight < 0:
        raise ValueError("cost loss weight must be non-negative and finite")
    if selection_loss_weight < 0 or decision_loss_weight < 0 or task_aux_weight < 0:
        raise ValueError("selection/decision/task-aux weights must be non-negative")
    if task_aux_weight > 0 and (task_aux_head is None or not task_label_map):
        raise ValueError("task auxiliary loss requires a head and a label map")
    router.train()
    if task_aux_head is not None:
        task_aux_head.train()
    total_accuracy_loss = 0.0
    total_accuracy_weight = 0.0
    total_supervised_pairs = 0
    total_cost_rank_loss = 0.0
    total_cost_rank_groups = 0
    total_cost_rank_pairs = 0
    total_selection_loss = 0.0
    total_selection_views = 0
    total_decision_loss = 0.0
    total_decision_views = 0
    total_task_aux_loss = 0.0
    total_task_aux_views = 0
    last_decision_lambda = 0.0

    for batch in data_loader:
        prompts, labels, _categories, costs = batch
        labels = labels.to(device)
        costs = costs.to(device)
        supervision_mask = getattr(
            batch,
            "supervision_mask",
            torch.ones_like(labels, dtype=torch.bool),
        ).to(device=device, dtype=torch.bool)
        routing_mask = getattr(
            batch,
            "routing_mask",
            torch.ones_like(labels, dtype=torch.bool),
        ).to(device=device, dtype=torch.bool)
        cost_available_mask = getattr(
            batch,
            "cost_available_mask",
            torch.ones_like(labels, dtype=torch.bool),
        ).to(device=device, dtype=torch.bool)
        routing_group_ids = getattr(
            batch,
            "routing_group_ids",
            [f"prompt:{prompt}" for prompt in prompts],
        )
        loss_targets = getattr(
            batch,
            "loss_targets",
            labels.to(dtype=torch.float32),
        ).to(device=device, dtype=torch.float32)
        loss_weights = getattr(
            batch,
            "loss_weights",
            torch.ones_like(labels, dtype=torch.float32),
        ).to(device=device, dtype=torch.float32)
        model_ids = model_ids.to(device)

        optimizer.zero_grad(set_to_none=True)
        prompt_embeddings = router.embedder(
            prompts,
            chunk_batch_size=getattr(router, "embedder_chunk_batch_size", 1),
            chunk_workers=getattr(router, "embedder_chunk_workers", 0),
        )
        logits = router._classify_prompt_embeddings(model_ids, prompt_embeddings, test_mode=False)
        loss_targets = loss_targets.to(dtype=logits.dtype)
        loss_weights = loss_weights.to(dtype=logits.dtype)
        accuracy_loss = supervised_cross_entropy(
            logits,
            labels,
            supervision_mask,
            loss_targets=loss_targets,
            loss_weights=loss_weights,
        )
        ranking = (
            cost_pairwise_ranking_loss(
                logits=logits,
                labels=labels,
                costs=costs,
                supervision_mask=supervision_mask,
                routing_mask=routing_mask,
                routing_group_ids=routing_group_ids,
                cost_available_mask=cost_available_mask,
                margin=cost_ranking_margin,
            )
            if cost_loss_weight > 0
            else CostRankingLoss(
                loss=logits.sum() * 0.0,
                eligible_group_count=0,
                pair_count=0,
            )
        )
        loss = accuracy_loss + cost_loss_weight * ranking.loss
        if selection_loss_weight > 0:
            selection_loss, selection_views = listwise_selection_loss(
                logits,
                labels.to(dtype=logits.dtype),
                supervision_mask,
                temperature=selection_temperature,
            )
            loss = loss + selection_loss_weight * selection_loss
            total_selection_loss += float(selection_loss.detach().item()) * selection_views
            total_selection_views += selection_views
        if decision_loss_weight > 0 and decision_lambda_sampler is not None:
            sampled_lambda = decision_lambda_sampler.next()
            decision_loss, decision_views = decision_aligned_loss(
                logits,
                labels.to(dtype=logits.dtype),
                costs,
                supervision_mask,
                routing_mask,
                cost_available_mask,
                temperature=selection_temperature,
                cost_weight=sampled_lambda,
            )
            loss = loss + decision_loss_weight * decision_loss
            total_decision_loss += float(decision_loss.detach().item()) * decision_views
            total_decision_views += decision_views
            last_decision_lambda = float(sampled_lambda)
        if task_aux_weight > 0:
            task_ids = torch.tensor(
                [task_label_map.get(str(category).split("/", 1)[0], -1) for category in _categories],
                dtype=torch.long,
                device=device,
            )
            valid = task_ids >= 0
            if bool(valid.any()):
                task_logits = task_aux_head(prompt_embeddings)
                task_loss = F.cross_entropy(task_logits[valid], task_ids[valid])
                loss = loss + task_aux_weight * task_loss
                valid_count = int(valid.sum().item())
                total_task_aux_loss += float(task_loss.detach().item()) * valid_count
                total_task_aux_views += valid_count
        loss.backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            clip_parameters = list(router.parameters())
            if task_aux_head is not None:
                clip_parameters += list(task_aux_head.parameters())
            torch.nn.utils.clip_grad_norm_(clip_parameters, gradient_clip_norm)
        optimizer.step()

        supervised_pair_count = int(supervision_mask.sum().item())
        supervised_weight = float(loss_weights[supervision_mask].sum().item())
        total_accuracy_loss += accuracy_loss.item() * supervised_weight
        total_accuracy_weight += supervised_weight
        total_supervised_pairs += supervised_pair_count
        total_cost_rank_loss += (
            ranking.loss.item() * ranking.eligible_group_count
        )
        total_cost_rank_groups += ranking.eligible_group_count
        total_cost_rank_pairs += ranking.pair_count

    if total_supervised_pairs == 0 or total_accuracy_weight <= 0:
        raise ValueError("training data loader produced no batches")
    accuracy_loss_value = total_accuracy_loss / total_accuracy_weight
    cost_rank_loss_value = (
        total_cost_rank_loss / total_cost_rank_groups
        if total_cost_rank_groups
        else 0.0
    )
    selection_loss_value = (
        total_selection_loss / total_selection_views if total_selection_views else 0.0
    )
    decision_loss_value = (
        total_decision_loss / total_decision_views if total_decision_views else 0.0
    )
    task_aux_loss_value = (
        total_task_aux_loss / total_task_aux_views if total_task_aux_views else 0.0
    )
    metrics = TrainingLossMetrics(
        total_loss=accuracy_loss_value + cost_loss_weight * cost_rank_loss_value
        + selection_loss_weight * selection_loss_value
        + decision_loss_weight * decision_loss_value
        + task_aux_weight * task_aux_loss_value,
        accuracy_loss=accuracy_loss_value,
        cost_rank_loss=cost_rank_loss_value,
        supervised_pair_count=total_supervised_pairs,
        cost_rank_group_count=total_cost_rank_groups,
        cost_rank_pair_count=total_cost_rank_pairs,
        selection_loss=selection_loss_value,
        selection_view_count=total_selection_views,
        decision_loss=decision_loss_value,
        decision_view_count=total_decision_views,
        decision_lambda=last_decision_lambda,
        task_aux_loss=task_aux_loss_value,
        task_aux_view_count=total_task_aux_views,
    )
    return metrics if return_metrics else metrics.total_loss


@torch.no_grad()
def evaluate_step_supervision(
    router: torch.nn.Module,
    data_loader: DataLoader,
    model_ids: torch.Tensor,
    device: torch.device,
) -> dict[str, float | int]:
    router.eval()
    weighted_loss_sum = 0.0
    weighted_absolute_error_sum = 0.0
    total_weight = 0.0
    pair_count = 0
    group_weights: dict[str, float] = defaultdict(float)

    for batch in data_loader:
        prompts, labels, _categories, _costs = batch
        supervision_mask = getattr(
            batch,
            "supervision_mask",
            torch.ones_like(labels, dtype=torch.bool),
        ).to(device=device, dtype=torch.bool)
        tail_offsets = getattr(batch, "tail_offsets", [None] * len(prompts))
        step_row_mask = torch.tensor(
            [tail_offset is not None for tail_offset in tail_offsets],
            device=device,
            dtype=torch.bool,
        )
        supervision_mask = supervision_mask & step_row_mask.unsqueeze(1)
        if not bool(supervision_mask.any()):
            continue

        targets = getattr(
            batch,
            "loss_targets",
            labels.to(dtype=torch.float32),
        ).to(device=device, dtype=torch.float32)
        weights = getattr(
            batch,
            "loss_weights",
            torch.ones_like(labels, dtype=torch.float32),
        ).to(device=device, dtype=torch.float32)
        logits = router(model_ids.to(device), prompts, test_mode=True)
        targets = targets.to(dtype=logits.dtype)
        weights = weights.to(dtype=logits.dtype)
        supervised_cross_entropy(
            logits,
            labels.to(device),
            supervision_mask,
            loss_targets=targets,
            loss_weights=weights,
        )

        margins = logits[:, :, 1] - logits[:, :, 0]
        selected_targets = targets[supervision_mask]
        selected_weights = weights[supervision_mask]
        terms = F.binary_cross_entropy_with_logits(
            margins[supervision_mask],
            selected_targets,
            reduction="none",
        )
        probabilities = torch.sigmoid(margins[supervision_mask])
        batch_weight = float(selected_weights.sum().item())
        weighted_loss_sum += float((terms * selected_weights).sum().item())
        weighted_absolute_error_sum += float(
            (torch.abs(probabilities - selected_targets) * selected_weights)
            .sum()
            .item()
        )
        total_weight += batch_weight
        pair_count += int(supervision_mask.sum().item())

        loss_group_ids = getattr(
            batch,
            "loss_group_ids",
            [f"prompt:{prompt}" for prompt in prompts],
        )
        for row_index, group_id in enumerate(loss_group_ids):
            if not bool(step_row_mask[row_index]):
                continue
            row_weight = float(
                weights[row_index][supervision_mask[row_index]].sum().item()
            )
            group_weights[str(group_id)] += row_weight

    if total_weight <= 0:
        return {
            "step_soft_loss": 0.0,
            "step_soft_mae": 0.0,
            "step_supervised_pair_count": 0,
            "step_trajectory_count": 0,
            "step_trajectory_weight_sum": 0.0,
        }
    return {
        "step_soft_loss": weighted_loss_sum / total_weight,
        "step_soft_mae": weighted_absolute_error_sum / total_weight,
        "step_supervised_pair_count": pair_count,
        "step_trajectory_count": len(group_weights),
        "step_trajectory_weight_sum": sum(group_weights.values()),
    }


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _select_inference_model(
    candidates: Mapping[int, Mapping[str, float | int | bool]],
    model_ids: Sequence[int],
    inference_cost_weight: float,
    category: str,
) -> int:
    if inference_cost_weight == 0:
        return max(
            model_ids,
            key=lambda model_id: (
                float(candidates[model_id]["score"]),
                -model_id,
            ),
        )

    candidate_costs: dict[int, float] = {}
    for model_id in model_ids:
        if not bool(candidates[model_id]["cost_available"]):
            raise ValueError(f"missing cost for category {category!r}, model_id={model_id}")
        cost = float(candidates[model_id]["cost"])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"invalid cost for category {category!r}, model_id={model_id}")
        candidate_costs[model_id] = cost

    minimum_cost = min(candidate_costs.values())
    cost_range = max(candidate_costs.values()) - minimum_cost

    def selection_key(model_id: int) -> tuple[float, float, float, int]:
        probability = _stable_sigmoid(float(candidates[model_id]["score"]))
        normalized_cost = (
            0.0
            if cost_range == 0
            else (candidate_costs[model_id] - minimum_cost) / cost_range
        )
        utility = probability - inference_cost_weight * normalized_cost
        return utility, probability, -candidate_costs[model_id], -model_id

    return max(model_ids, key=selection_key)


@torch.no_grad()
def evaluate_router(
    router: torch.nn.Module,
    data_loader: DataLoader,
    model_ids: torch.Tensor,
    device: torch.device,
    model_names: Mapping[int, str] | None = None,
    include_benchmark_costs: bool = False,
    inference_cost_weight: float = 0.0,
) -> dict[str, Any]:
    if (
        isinstance(inference_cost_weight, bool)
        or not isinstance(inference_cost_weight, (int, float))
        or not math.isfinite(float(inference_cost_weight))
        or inference_cost_weight < 0
    ):
        raise ValueError("inference cost weight must be non-negative and finite")
    inference_cost_weight = float(inference_cost_weight)
    router.eval()
    model_names = model_names or {}
    category_successes: dict[str, int] = defaultdict(int)
    category_counts: dict[str, int] = defaultdict(int)
    model_counts: Counter[str] = Counter()
    benchmark_costs: dict[str, float] = defaultdict(float)
    total_successes = 0
    total_prompts = 0
    total_cost = 0.0
    multi_correct_prompts = 0
    multi_correct_selected_successes = 0
    cheapest_correct_hits = 0
    correct_cost_regret_total = 0.0
    routing_groups: dict[str, dict[str, Any]] = {}
    expected_model_ids = [int(model_id) for model_id in model_ids.detach().cpu().tolist()]

    for batch in data_loader:
        prompts, labels, categories, costs = batch
        routing_mask = getattr(
            batch,
            "routing_mask",
            torch.ones_like(labels, dtype=torch.bool),
        ).to(dtype=torch.bool)
        cost_available_mask = getattr(
            batch,
            "cost_available_mask",
            torch.ones_like(labels, dtype=torch.bool),
        ).to(dtype=torch.bool)
        routing_group_ids = getattr(
            batch,
            "routing_group_ids",
            [f"prompt:{prompt}" for prompt in prompts],
        )
        active_indices = (
            torch.nonzero(routing_mask.any(dim=1), as_tuple=False)
            .flatten()
            .tolist()
        )
        if not active_indices:
            continue

        active_prompts = [prompts[index] for index in active_indices]
        active_categories = [categories[index] for index in active_indices]
        active_group_ids = [routing_group_ids[index] for index in active_indices]
        labels = labels.to(device)
        costs = costs.to(device)
        routing_mask = routing_mask.to(device)
        cost_available_mask = cost_available_mask.to(device)
        model_ids = model_ids.to(device)
        active_index_tensor = torch.tensor(
            active_indices,
            dtype=torch.long,
            device=device,
        )
        active_labels = labels.index_select(0, active_index_tensor)
        active_costs = costs.index_select(0, active_index_tensor)
        active_routing_mask = routing_mask.index_select(0, active_index_tensor)
        active_cost_available_mask = cost_available_mask.index_select(
            0, active_index_tensor
        )
        logits = router(model_ids, active_prompts, test_mode=True)
        if logits.shape[:2] != (len(active_prompts), len(expected_model_ids)):
            raise ValueError(
                "router logits do not match active prompt/model dimensions: "
                f"{tuple(logits.shape)}"
            )
        success_margins = logits[:, :, 1] - logits[:, :, 0]

        for row_index, (category, group_id) in enumerate(
            zip(active_categories, active_group_ids)
        ):
            group = routing_groups.setdefault(
                str(group_id),
                {
                    "category": str(category),
                    "candidates": {},
                },
            )
            if group["category"] != str(category):
                raise ValueError(
                    f"routing group {group_id!r} contains inconsistent categories"
                )
            candidates: dict[int, dict[str, float | int]] = group["candidates"]
            for logit_index, model_id in enumerate(expected_model_ids):
                if model_id < 0 or model_id >= active_routing_mask.size(1):
                    raise ValueError(
                        f"model_id={model_id} is outside routing mask width "
                        f"{active_routing_mask.size(1)}"
                    )
                if not bool(active_routing_mask[row_index, model_id]):
                    continue
                if model_id in candidates:
                    raise ValueError(
                        f"routing group {group_id!r} has duplicate canonical "
                        f"view for model_id={model_id}"
                    )
                candidates[model_id] = {
                    "score": float(success_margins[row_index, logit_index].item()),
                    "label": int(active_labels[row_index, model_id].item()),
                    "cost": float(active_costs[row_index, model_id].item()),
                    "cost_available": bool(
                        active_cost_available_mask[row_index, model_id]
                    ),
                }

    expected_model_id_set = set(expected_model_ids)
    for group_id in sorted(routing_groups):
        group = routing_groups[group_id]
        category = str(group["category"])
        candidates = group["candidates"]
        missing_model_ids = expected_model_id_set - set(candidates)
        if missing_model_ids:
            raise ValueError(
                f"routing group {group_id!r} is missing canonical views for "
                f"model_ids={sorted(missing_model_ids)}"
            )
        selected_model_id = _select_inference_model(
            candidates=candidates,
            model_ids=expected_model_ids,
            inference_cost_weight=inference_cost_weight,
            category=category,
        )
        success = int(candidates[selected_model_id]["label"])
        selected_cost_value = float(candidates[selected_model_id]["cost"])
        if not math.isfinite(selected_cost_value) or selected_cost_value < 0:
            raise ValueError(f"invalid selected cost for category {category!r}")
        correct_model_ids = [
            model_id
            for model_id in expected_model_ids
            if int(candidates[model_id]["label"]) == 1
        ]
        if len(correct_model_ids) >= 2:
            correct_costs = {
                model_id: float(candidates[model_id]["cost"])
                for model_id in correct_model_ids
            }
            for model_id, cost in correct_costs.items():
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError(
                        f"invalid correct-model cost for category {category!r}, "
                        f"model_id={model_id}"
                    )
            minimum_correct_cost = min(correct_costs.values())
            cheapest_correct_model_ids = {
                model_id
                for model_id, cost in correct_costs.items()
                if math.isclose(
                    cost,
                    minimum_correct_cost,
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                )
            }
            multi_correct_prompts += 1
            if success:
                multi_correct_selected_successes += 1
                correct_cost_regret_total += max(
                    0.0,
                    selected_cost_value - minimum_correct_cost,
                )
                if selected_model_id in cheapest_correct_model_ids:
                    cheapest_correct_hits += 1

        category_successes[category] += success
        category_counts[category] += 1
        total_successes += success
        total_prompts += 1
        total_cost += selected_cost_value
        if include_benchmark_costs:
            benchmark = category.split("/", 1)[0]
            benchmark_costs[benchmark] += selected_cost_value
        model_counts[
            model_names.get(selected_model_id, str(selected_model_id))
        ] += 1

    if total_prompts == 0:
        raise ValueError("evaluation data loader produced no batches")

    category_accuracy = {
        category: category_successes[category] / category_counts[category]
        for category in sorted(category_counts)
    }
    dataset_accuracy = sum(category_accuracy.values()) / len(category_accuracy)

    metrics = {
        "sample_accuracy": total_successes / total_prompts,
        "dataset_accuracy": dataset_accuracy,
        "category_accuracy": category_accuracy,
        "model_counts": dict(model_counts),
        "total_cost": total_cost,
        "mean_selected_cost": total_cost / total_prompts,
        "total_prompts": total_prompts,
        "total_successes": total_successes,
        "multi_correct_prompts": multi_correct_prompts,
        "correct_cost_regret_prompts": multi_correct_selected_successes,
        "cheapest_correct_hit_rate": (
            cheapest_correct_hits / multi_correct_prompts
            if multi_correct_prompts
            else 0.0
        ),
        "conditional_cheapest_correct_rate": (
            cheapest_correct_hits / multi_correct_selected_successes
            if multi_correct_selected_successes
            else 0.0
        ),
        "mean_correct_cost_regret": (
            correct_cost_regret_total / multi_correct_selected_successes
            if multi_correct_selected_successes
            else 0.0
        ),
        "inference_cost_weight": inference_cost_weight,
    }
    if include_benchmark_costs:
        return {
            **metrics,
            "cost_by_benchmark": dict(sorted(benchmark_costs.items())),
        }
    return metrics


def should_run_train_eval(epoch: int, train_eval_every: int) -> bool:
    return train_eval_every > 0 and epoch % train_eval_every == 0


def prefix_evaluation_metrics(metrics: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def format_epoch_metrics(metrics: Mapping[str, Any]) -> str:
    parts = [f"epoch={int(metrics['epoch'])}", f"train_loss={float(metrics['train_loss']):.6f}"]
    if "train_accuracy_loss" in metrics:
        parts.append(
            f"train_accuracy_loss={float(metrics['train_accuracy_loss']):.6f}"
        )
    if "train_cost_rank_loss" in metrics:
        parts.append(
            f"train_cost_rank_loss={float(metrics['train_cost_rank_loss']):.6f}"
        )
    if "effective_cost_loss_weight" in metrics:
        parts.append(
            f"cost_weight={float(metrics['effective_cost_loss_weight']):.6f}"
        )
    if "train_loader_mode" in metrics:
        parts.append(f"train_loader={metrics['train_loader_mode']}")
    if "dataset_accuracy" in metrics:
        parts.append(
            f"validation_dataset_acc={float(metrics['dataset_accuracy']):.4f}"
        )
    if "sample_accuracy" in metrics:
        parts.append(
            f"validation_sample_acc={float(metrics['sample_accuracy']):.4f}"
        )
    if "step_soft_loss" in metrics:
        parts.append(
            f"validation_step_soft_loss={float(metrics['step_soft_loss']):.6f}"
        )
    if "step_soft_mae" in metrics:
        parts.append(
            f"validation_step_soft_mae={float(metrics['step_soft_mae']):.6f}"
        )
    if "train_dataset_accuracy" in metrics:
        parts.append(f"train_dataset_acc={float(metrics['train_dataset_accuracy']):.4f}")
    if "train_sample_accuracy" in metrics:
        parts.append(f"train_sample_acc={float(metrics['train_sample_accuracy']):.4f}")
    return " ".join(parts)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoutingGroupBatchSampler(Sampler[list[int]]):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        generator: torch.Generator | None = None,
        max_group_size: int = 64,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_group_size <= 0:
            raise ValueError("max_group_size must be positive")
        items = getattr(dataset, "items", None)
        if items is None:
            raise TypeError("routing-group batching requires a dataset with items")

        grouped_indices: dict[str, list[int]] = {}
        for index, item in enumerate(items):
            routing_mask = getattr(item, "routing_mask", None)
            has_canonical_candidate = (
                routing_mask is None or bool(routing_mask.any())
            )
            if has_canonical_candidate:
                group_id = str(
                    getattr(item, "routing_group_id", "")
                    or getattr(item, "prompt_id", "")
                    or f"index:{index}"
                )
            else:
                group_id = f"noncanonical:{index}"
            grouped_indices.setdefault(group_id, []).append(index)

        oversized_groups = {
            group_id: len(indices)
            for group_id, indices in grouped_indices.items()
            if len(indices) > max_group_size
        }
        if oversized_groups:
            group_id, group_size = max(
                oversized_groups.items(),
                key=lambda item: item[1],
            )
            raise ValueError(
                f"routing group {group_id!r} contains {group_size} views, exceeding "
                f"--max-routing-group-views={max_group_size}; reduce expansion or "
                "raise the limit explicitly if the resulting memory use is acceptable"
            )

        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.generator = generator
        batches: list[tuple[int, ...]] = []
        current_batch: list[int] = []
        for group in grouped_indices.values():
            if current_batch and len(current_batch) + len(group) > self.batch_size:
                batches.append(tuple(current_batch))
                current_batch = []
            current_batch.extend(group)
            if len(current_batch) >= self.batch_size:
                batches.append(tuple(current_batch))
                current_batch = []
        if current_batch:
            batches.append(tuple(current_batch))
        self.batches = tuple(batches)

    def _ordered_batches(self) -> list[tuple[int, ...]]:
        if not self.shuffle or len(self.batches) < 2:
            return list(self.batches)
        order = torch.randperm(
            len(self.batches),
            generator=self.generator,
        ).tolist()
        return [self.batches[index] for index in order]

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self._ordered_batches():
            yield list(batch)

    def __len__(self) -> int:
        return len(self.batches)


def build_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator | None = None,
    preserve_routing_groups: bool = False,
    max_routing_group_views: int = 64,
) -> DataLoader:
    if preserve_routing_groups:
        return DataLoader(
            dataset,
            batch_sampler=RoutingGroupBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=generator,
                max_group_size=max_routing_group_views,
            ),
            num_workers=num_workers,
            collate_fn=collate_prompt_batch,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_prompt_batch,
        generator=generator,
    )


def compute_window_token_lengths(
    token_count: int,
    max_chunk_length: int,
    overlap_tokens: int,
) -> list[int]:
    if max_chunk_length <= 0:
        raise ValueError("max_chunk_length must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_chunk_length:
        raise ValueError("overlap_tokens must be in [0, max_chunk_length)")
    if token_count <= max_chunk_length:
        return [token_count]

    stride = max_chunk_length - overlap_tokens
    lengths: list[int] = []
    for start_idx in range(0, token_count, stride):
        end_idx = min(start_idx + max_chunk_length, token_count)
        lengths.append(end_idx - start_idx)
        if end_idx >= token_count:
            break
    return lengths


def build_shape_probe_summary_from_token_lengths(
    token_lengths: Sequence[int],
    max_chunk_length: int,
    overlap_tokens: int,
    embedding_dim: int,
    num_models: int,
) -> dict[str, Any]:
    if not token_lengths:
        raise ValueError("shape probe requires at least one prompt")

    chunk_token_lengths = [
        compute_window_token_lengths(token_count, max_chunk_length, overlap_tokens)
        for token_count in token_lengths
    ]
    chunk_counts = [len(lengths) for lengths in chunk_token_lengths]
    batch_size = len(token_lengths)
    max_chunks = max(chunk_counts)
    total_chunks = sum(chunk_counts)
    stride = max_chunk_length - overlap_tokens
    loss_item_count = batch_size * num_models

    return {
        "batch_size": batch_size,
        "window_size": max_chunk_length,
        "overlap_tokens": overlap_tokens,
        "stride": stride,
        "prompt_token_lengths": list(token_lengths),
        "chunk_counts": chunk_counts,
        "chunk_token_lengths": chunk_token_lengths,
        "padded_chunk_length": max_chunk_length,
        "flat_input_ids_shape": (total_chunks, max_chunk_length),
        "chunk_embeddings_shape": (batch_size, max_chunks, embedding_dim),
        "attention_mask_shape": (batch_size, max_chunks, max_chunks),
        "prompt_embedding_shape": (batch_size, embedding_dim),
        "model_embedding_shape": (num_models, embedding_dim),
        "sim_matrix_shape": (batch_size, num_models, embedding_dim),
        "logits_shape": (batch_size, num_models, 2),
        "labels_shape": (batch_size, num_models),
        "loss_logits_shape": (loss_item_count, 2),
        "loss_labels_shape": (loss_item_count,),
        "loss_shape": (),
    }


def _shape_tuple(value: Any) -> str:
    return str(tuple(value)) if isinstance(value, (tuple, list)) else str(value)


def collect_shape_probe_summary(
    router: torch.nn.Module,
    items: Sequence[PromptBatchItem],
    num_models: int,
) -> dict[str, Any]:
    embedding_model = router.embedder.embedding_model
    prompts = [item.prompt for item in items]
    prompt_ids = [item.prompt_id for item in items]
    categories = [item.category for item in items]
    prompt_char_lengths = [len(prompt) for prompt in prompts]
    token_lengths: list[int] = []
    chunk_token_lengths: list[list[int]] = []

    for prompt in prompts:
        encoding = embedding_model.tokenize(prompt)
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        token_lengths.append(int(input_ids.shape[0]))
        chunks = embedding_model.chunk(input_ids, attention_mask)
        chunk_token_lengths.append(
            [int(chunk["attention_mask"].sum().item()) for chunk in chunks]
        )

    summary = build_shape_probe_summary_from_token_lengths(
        token_lengths=token_lengths,
        max_chunk_length=int(embedding_model.max_chunk_length),
        overlap_tokens=int(embedding_model.overlap_tokens),
        embedding_dim=int(router.text_dim),
        num_models=num_models,
    )
    summary["prompt_ids"] = prompt_ids
    summary["categories"] = categories
    summary["prompt_char_lengths"] = prompt_char_lengths
    summary["chunk_token_lengths"] = chunk_token_lengths
    summary["chunk_counts"] = [len(lengths) for lengths in chunk_token_lengths]
    summary["flat_input_ids_shape"] = (
        sum(summary["chunk_counts"]),
        int(embedding_model.max_chunk_length),
    )
    summary["chunk_embeddings_shape"] = (
        len(items),
        max(summary["chunk_counts"]),
        int(router.text_dim),
    )
    summary["attention_mask_shape"] = (
        len(items),
        max(summary["chunk_counts"]),
        max(summary["chunk_counts"]),
    )
    return summary


def print_shape_probe_summary(summary: Mapping[str, Any]) -> None:
    print("[shape-probe] LongContextRouter dimensions")
    print(f"[shape-probe] batch_size = {summary['batch_size']}")
    print(f"[shape-probe] prompt_ids = {summary['prompt_ids']}")
    print(f"[shape-probe] categories = {summary['categories']}")
    print(f"[shape-probe] prompt_char_lengths = {summary['prompt_char_lengths']}")
    print(f"[shape-probe] prompt_token_lengths = {summary['prompt_token_lengths']}")
    print(f"[shape-probe] window_size/max_chunk_length = {summary['window_size']}")
    print(f"[shape-probe] overlap_tokens = {summary['overlap_tokens']}")
    print(f"[shape-probe] stride = {summary['stride']}")
    print(f"[shape-probe] chunk_counts = {summary['chunk_counts']}")
    print(f"[shape-probe] chunk_token_lengths = {summary['chunk_token_lengths']}")
    print(f"[shape-probe] padded_chunk_length = {summary['padded_chunk_length']}")
    for key in [
        "flat_input_ids_shape",
        "chunk_embeddings_shape",
        "attention_mask_shape",
        "prompt_embedding_shape",
        "model_embedding_shape",
        "sim_matrix_shape",
        "logits_shape",
        "labels_shape",
        "loss_logits_shape",
        "loss_labels_shape",
        "loss_shape",
    ]:
        print(f"[shape-probe] {key} = {_shape_tuple(summary[key])}")


def run_shape_probe(
    router: torch.nn.Module,
    dataset: LongContextRouterDataset,
    batch_size: int,
    num_models: int,
) -> dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError("shape probe requires a non-empty dataset")
    probe_size = max(1, min(batch_size, len(dataset)))
    items = sorted(dataset.items, key=lambda item: len(item.prompt), reverse=True)[:probe_size]
    summary = collect_shape_probe_summary(router, items, num_models=num_models)
    print_shape_probe_summary(summary)
    print("[shape-probe] note = tokenizer/chunk shape probe only; Qwen forward/backward was not run")
    return summary


def build_router(
    args: argparse.Namespace,
    num_models: int,
    device: torch.device,
    use_embedding_cache: bool = True,
) -> torch.nn.Module:
    from baselines.LongContextRouter.model import LongContextRouter

    cache_path = getattr(args, "embedding_cache_path", None) if use_embedding_cache else None
    return LongContextRouter(
        embedder_model_name=args.embedder_model_name,
        embedder_max_chunk_length_k=args.embedder_max_chunk_length_k,
        embedder_overlap_tokens_k=args.embedder_overlap_tokens_k,
        embedder_aggregator_n_layers=args.embedder_aggregator_n_layers,
        embedder_aggregator_dropout=args.embedder_aggregator_dropout,
        embedder_qk_norm=not args.no_embedder_qk_norm,
        embedder_rope_base=args.embedder_rope_base,
        embedder_chunk_batch_size=args.embedder_chunk_batch_size,
        embedder_chunk_workers=args.embedder_chunk_workers,
        embedder_backend=args.embedder_backend,
        embedding_cache_path=str(cache_path) if cache_path else None,
        fusion_mode=args.fusion_mode,
        alpha=args.alpha,
        num_LLM_models=num_models,
        device=str(device),
    )


def embedding_cache_config(args: argparse.Namespace, embedding_dim: int) -> dict[str, Any]:
    return {
        "model_name": str(args.embedder_model_name),
        "backend": str(args.embedder_backend),
        "max_chunk_length": int(args.embedder_max_chunk_length_k * 1024),
        "overlap_tokens": int(args.embedder_overlap_tokens_k * 1024),
        "embedding_dim": int(embedding_dim),
    }


@torch.no_grad()
def precompute_prompt_chunk_embeddings(
    router: torch.nn.Module,
    datasets: Sequence[LongContextRouterDataset],
    cache_path: str | Path,
    args: argparse.Namespace,
) -> None:
    from baselines.LongContextRouter.embedding_cache import ChunkEmbeddingCache

    embedding_model = router.embedder.embedding_model
    if embedding_model is None:
        raise RuntimeError("embedding cache builder requires a raw embedding backend")

    prompts = list(
        dict.fromkeys(item.prompt for dataset in datasets for item in dataset.items)
    )
    if not prompts:
        raise ValueError("cannot build an embedding cache without prompts")

    embedding_model.eval()
    chunk_embeddings_by_prompt: dict[str, torch.Tensor] = {}
    for start in range(0, len(prompts), args.batch_size):
        prompt_batch = prompts[start : start + args.batch_size]
        chunk_embeddings, _attention_mask, chunk_counts = embedding_model(
            prompt_batch,
            chunk_batch_size=args.embedder_chunk_batch_size,
            chunk_workers=args.embedder_chunk_workers,
        )
        for index, (prompt, count) in enumerate(zip(prompt_batch, chunk_counts)):
            with torch.inference_mode(False):
                cached = (
                    chunk_embeddings[index, :count]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                    .clone()
                )
            chunk_embeddings_by_prompt[prompt] = cached
        completed = min(start + len(prompt_batch), len(prompts))
        print(f"[embedding-cache] encoded {completed}/{len(prompts)} prompts")

    cache = ChunkEmbeddingCache.build(
        prompts=prompts,
        chunk_embeddings_by_prompt=chunk_embeddings_by_prompt,
        config=embedding_cache_config(args, embedding_dim=router.text_dim),
    )
    cache.save(cache_path)
    print(f"[embedding-cache] wrote {cache_path}")


def validate_router_embedding_cache(
    router: torch.nn.Module,
    datasets: Sequence[LongContextRouterDataset],
) -> None:
    embedding_cache = getattr(router.embedder, "embedding_cache", None)
    if embedding_cache is None:
        return None
    prompts = list(
        dict.fromkeys(item.prompt for dataset in datasets for item in dataset.items)
    )
    router.embedder.validate_cached_prompts(prompts)


def load_state_dict_flexibly(
    router: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    path_class = type(Path())
    with torch.serialization.safe_globals([path_class]):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, Mapping)
        else checkpoint
    )
    if not isinstance(state_dict, Mapping):
        raise ValueError(
            f"checkpoint must be a state-dict mapping: {checkpoint_path}"
        )
    router.load_state_dict(state_dict)




def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LongContextRouter end-to-end on raw prompts")
    parser.add_argument("--train-data-path", default=None, help="EmbedLLM-style train_ours.csv")
    parser.add_argument("--test-data-path", default=None, help="EmbedLLM-style test_ours.csv")
    parser.add_argument("--swe-agent-root", type=Path, default=DEFAULT_SWE_AGENT_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model slugs under <swe-agent-root>/sweagent_results for direct prompt extraction.",
    )
    parser.add_argument(
        "--benches",
        nargs="+",
        choices=list(SUPPORTED_BENCHES),
        default=list(DEFAULT_DIRECT_BENCHES),
        help="Benchmarks to use when extracting prompts directly from SWE-agent data.",
    )
    parser.add_argument(
        "--exclude-training-benches",
        nargs="+",
        choices=list(SUPPORTED_BENCHES),
        default=[],
        help="Direct SWE-agent only: exclude benches from train/internal validation after the original split and SWE expansion; test is unchanged.",
    )
    parser.add_argument("--source-split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Result run directory under each bench. Default: <source-split>_split.",
    )
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--limit-prompts-per-bench", type=int, default=None)
    parser.add_argument("--fill-missing-zero", action="store_true")
    parser.add_argument(
        "--swe-bench-positive-tail-steps",
        type=int,
        default=0,
        help=(
            "Cache-backed SWE-bench Pro expansion: each positive internal-train "
            "model record uses this many trailing actionable steps; negative "
            "train and all internal-validation model records use only the final "
            "step. Zero preserves legacy prompt construction."
        ),
    )
    parser.add_argument(
        "--swe-bench-step-supervision-mode",
        choices=list(SWE_BENCH_STEP_SUPERVISION_MODES),
        default=None,
        help=(
            "Explicit SWE-bench Pro step supervision. single_hard uses only the "
            "final actionable step with the original 0/1 target; all_steps_soft "
            "uses every actionable step with target=hard_label/tail_offset and "
            "per-trajectory weight=1/T. Omit to retain legacy behavior."
        ),
    )
    parser.add_argument(
        "--swe-bench-training-prompt-mode",
        choices=list(SWE_BENCH_TRAINING_PROMPT_MODES),
        default="legacy",
        help=(
            "Training prompt construction when SWE step expansion is disabled. "
            "legacy includes historical model outputs; problem_only uses only "
            "the problem statement. The default preserves existing behavior."
        ),
    )
    parser.add_argument("--ifbench-mode", choices=["loose", "strict"], default="loose")
    parser.add_argument(
        "--memory-label",
        choices=["f1_threshold", "exact_match", "substring_exact_match"],
        default="f1_threshold",
    )
    parser.add_argument("--memory-f1-threshold", type=float, default=0.5)
    parser.add_argument(
        "--memory-benchmark-version",
        choices=["legacy", "128k_v2", "128k_v2_tem0"],
        default="legacy",
        help="Select matching MemoryAgentBench splits, results, and costs.",
    )

    parser.add_argument("--cost-source", choices=["auto", "trace", "traj", "none"], default="auto")
    parser.add_argument(
        "--cost-overrides-path", type=Path, default=None,
        help="Validated per-split/model/prompt billing overlay; original benchmark artifacts remain unchanged.",
    )
    parser.add_argument("--require-cost", action="store_true")
    parser.add_argument("--model-save-path", default="baselines/LongContextRouter/data/saved_model.pth")
    parser.add_argument("--best-model-save-path", default=None)
    parser.add_argument("--best-accuracy-model-save-path", default=None)
    parser.add_argument("--best-cost-aware-model-save-path", default=None)
    parser.add_argument("--last-model-save-path", default=None)
    parser.add_argument("--model-load-path", default=None)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume model, optimizer, history, RNG, and epoch from a saved last checkpoint.",
    )
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--model-num", type=int, default=None)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--cost-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for ranking cheaper models above costlier models when both "
            "answered the same prompt correctly."
        ),
    )
    parser.add_argument(
        "--cost-ranking-margin",
        type=float,
        default=0.1,
        help="Required success-margin gap between cheaper and costlier correct models.",
    )
    parser.add_argument(
        "--cost-loss-warmup-epochs",
        type=int,
        default=0,
        help="Train with correctness loss only for this many initial epochs.",
    )
    parser.add_argument(
        "--selection-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the listwise top-1 selection loss (correct-model probability mass).",
    )
    parser.add_argument(
        "--selection-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to success margins in the selection/decision losses.",
    )
    parser.add_argument(
        "--decision-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the differentiable cost-aware decision loss.",
    )
    parser.add_argument(
        "--decision-lambdas",
        type=float,
        nargs="+",
        default=(0.0, 0.1, 0.3, 1.0),
        help="Cost weights sampled per batch for the decision loss.",
    )
    parser.add_argument(
        "--task-aux-weight",
        type=float,
        default=0.0,
        help="Weight for the benchmark-classification auxiliary loss.",
    )
    parser.add_argument(
        "--task-aux-classes",
        type=int,
        default=0,
        help="Benchmark classes for the auxiliary head (0 disables the head).",
    )
    parser.add_argument(
        "--routing-group-sampler-mode",
        choices=["standard", "grouped", "after_warmup"],
        default="standard",
        help=(
            "Training sampler schedule: standard never preserves groups, grouped "
            "always preserves them, and after_warmup switches after the cost-loss "
            "warmup epochs."
        ),
    )
    parser.add_argument(
        "--max-routing-group-views",
        type=int,
        default=64,
        help=(
            "Reject an expanded routing group above this many views before it "
            "can create an oversized cost-aware batch."
        ),
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--train-eval-every",
        type=int,
        default=1,
        help="Evaluate train accuracy every N epochs. Use 0 to disable train accuracy evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed", type=int, default=None,
        help="Keep the direct-source train/validation split fixed while changing training RNG; defaults to --seed.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedder-model-name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedder-max-chunk-length-k", type=float, default=8.0)
    parser.add_argument("--embedder-overlap-tokens-k", type=float, default=1.0)
    parser.add_argument("--embedder-aggregator-n-layers", type=int, default=1)
    parser.add_argument(
        "--embedder-aggregator-dropout",
        type=float,
        default=0.0,
        help="Dropout probability for aggregator attention weights and SwiGLU activations.",
    )
    parser.add_argument("--no-embedder-qk-norm", action="store_true")
    parser.add_argument("--embedder-rope-base", type=float, default=10000.0)
    parser.add_argument("--embedder-chunk-batch-size", type=int, default=4)
    parser.add_argument("--embedder-chunk-workers", type=int, default=0)
    parser.add_argument(
        "--embedding-cache-path",
        type=Path,
        default=None,
        help="Frozen per-chunk embedding cache. Training loads this without constructing Qwen/vLLM.",
    )
    parser.add_argument(
        "--build-embedding-cache-only",
        action="store_true",
        help="Build --embedding-cache-path from the selected train/eval prompts and exit.",
    )
    parser.add_argument(
        "--embedder-backend",
        choices=["transformers", "sentence_transformers", "sentence-transformers", "vllm"],
        default="transformers",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["product", "sigmoid_p_mul_q", "p_mul_sigmoid_q"],
        default="product",
        help="Elementwise fusion applied to model embedding p and prompt embedding q.",
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--shape-probe-only",
        action="store_true",
        help="Print longest-batch LongContextRouter tensor dimensions and exit before training.",
    )
    parser.add_argument(
        "--save-best-metric",
        choices=["dataset_accuracy", "sample_accuracy"],
        default="sample_accuracy",
    )
    return parser.parse_args(argv)


def resolve_swe_step_supervision_mode(args: argparse.Namespace) -> str | None:
    mode = getattr(args, "swe_bench_step_supervision_mode", None)
    if mode is None:
        return None
    normalized = str(mode).strip()
    if normalized not in SWE_BENCH_STEP_SUPERVISION_MODES:
        raise ValueError(f"unsupported SWE-bench step supervision mode: {mode!r}")
    return normalized


def validate_swe_bench_embedding_expansion(args: argparse.Namespace) -> None:
    positive_tail_steps = int(args.swe_bench_positive_tail_steps)
    step_supervision_mode = resolve_swe_step_supervision_mode(args)
    training_prompt_mode = str(
        getattr(args, "swe_bench_training_prompt_mode", "legacy")
    )
    if training_prompt_mode not in SWE_BENCH_TRAINING_PROMPT_MODES:
        raise ValueError(
            f"unsupported SWE-bench training prompt mode: {training_prompt_mode!r}"
        )
    if positive_tail_steps < 0:
        raise ValueError("--swe-bench-positive-tail-steps must be non-negative")
    if (
        training_prompt_mode == "problem_only"
        and (step_supervision_mode is not None or positive_tail_steps > 0)
    ):
        raise ValueError(
            "SWE-bench problem_only training cannot be combined with step expansion"
        )
    if step_supervision_mode is not None and positive_tail_steps != 0:
        raise ValueError(
            "--swe-bench-step-supervision-mode and "
            "--swe-bench-positive-tail-steps are mutually exclusive"
        )
    if positive_tail_steps == 0 and step_supervision_mode is None:
        return
    option = (
        "--swe-bench-step-supervision-mode"
        if step_supervision_mode is not None
        else "--swe-bench-positive-tail-steps"
    )
    if args.embedding_cache_path is None:
        raise ValueError(f"{option} requires --embedding-cache-path")
    if not args.models:
        raise ValueError(f"{option} requires direct SWE-agent data via --models")
    if "swe_bench_pro" not in args.benches:
        raise ValueError(
            f"{option} requires --benches to include swe_bench_pro"
        )


def validate_output_paths(args: argparse.Namespace) -> None:
    output_specs = (
        ("--model-save-path", args.model_save_path, ".pth"),
        ("--best-model-save-path", args.best_model_save_path, ".pth"),
        (
            "--best-accuracy-model-save-path",
            args.best_accuracy_model_save_path,
            ".pth",
        ),
        (
            "--best-cost-aware-model-save-path",
            args.best_cost_aware_model_save_path,
            ".pth",
        ),
        ("--last-model-save-path", args.last_model_save_path, ".pth"),
        ("--metrics-output", args.metrics_output, ".json"),
    )
    for argument, value, required_suffix in output_specs:
        if value is None:
            continue
        path = Path(value)
        if path.suffix.lower() != required_suffix:
            raise ValueError(f"{argument} must end in {required_suffix}")
        if path.exists() and path.is_dir():
            raise ValueError(f"{argument} must be a file path, got directory: {path}")
        existing_parent = path.parent
        while not existing_parent.exists() and existing_parent != existing_parent.parent:
            existing_parent = existing_parent.parent
        if existing_parent.exists() and not existing_parent.is_dir():
            raise ValueError(
                f"{argument} parent path is not a directory: {existing_parent}"
            )

    model_save_path = Path(args.model_save_path)
    checkpoint_paths = (
        ("--model-save-path", model_save_path),
        (
            "--best-model-save-path",
            Path(args.best_model_save_path)
            if args.best_model_save_path
            else default_best_model_path(model_save_path),
        ),
        (
            "--best-accuracy-model-save-path",
            Path(args.best_accuracy_model_save_path)
            if args.best_accuracy_model_save_path
            else default_best_accuracy_model_path(model_save_path),
        ),
        (
            "--best-cost-aware-model-save-path",
            Path(args.best_cost_aware_model_save_path)
            if args.best_cost_aware_model_save_path
            else default_best_cost_aware_model_path(model_save_path),
        ),
        (
            "--last-model-save-path",
            Path(args.last_model_save_path)
            if args.last_model_save_path
            else default_last_model_path(model_save_path),
        ),
    )
    seen_paths: dict[Path, str] = {}
    for argument, path in checkpoint_paths:
        resolved_path = path.resolve()
        conflicting_argument = seen_paths.get(resolved_path)
        if conflicting_argument is not None:
            raise ValueError(
                f"{argument} must not overwrite {conflicting_argument}: {path}"
            )
        seen_paths[resolved_path] = argument


def validate_cost_aware_datasets(
    datasets: Sequence[LongContextRouterDataset],
) -> None:
    for dataset in datasets:
        for item in dataset.items:
            supervision_mask = (
                item.supervision_mask
                if item.supervision_mask is not None
                else torch.ones_like(item.labels, dtype=torch.bool)
            )
            routing_mask = (
                item.routing_mask
                if item.routing_mask is not None
                else torch.ones_like(item.labels, dtype=torch.bool)
            )
            cost_available_mask = (
                item.cost_available_mask
                if item.cost_available_mask is not None
                else torch.ones_like(item.labels, dtype=torch.bool)
            )
            required_mask = supervision_mask & routing_mask
            missing_model_ids = torch.nonzero(
                required_mask & ~cost_available_mask,
                as_tuple=False,
            ).flatten().tolist()
            if missing_model_ids:
                raise ValueError(
                    f"cost-aware training requires explicit costs for prompt "
                    f"{item.prompt_id!r}, model_ids={missing_model_ids}"
                )
            required_costs = item.costs[required_mask]
            if required_costs.numel() and (
                not bool(torch.isfinite(required_costs).all())
                or bool((required_costs < 0).any())
            ):
                raise ValueError(
                    f"cost-aware training found invalid cost for prompt "
                    f"{item.prompt_id!r}"
                )


def effective_cost_loss_weight(
    configured_weight: float,
    warmup_epochs: int,
    epoch: int,
) -> float:
    return 0.0 if epoch <= warmup_epochs else configured_weight


def select_training_loader(
    *,
    standard_loader: DataLoader,
    grouped_loader: DataLoader,
    sampler_mode: str,
    epoch: int,
    warmup_epochs: int,
) -> DataLoader:
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if sampler_mode == "standard":
        return standard_loader
    if sampler_mode == "grouped":
        return grouped_loader
    if sampler_mode == "after_warmup":
        return (
            standard_loader
            if epoch <= warmup_epochs
            else grouped_loader
        )
    raise ValueError(f"unsupported routing group sampler mode: {sampler_mode}")


def _dataset_has_sparse_routing_views(dataset: Dataset) -> bool:
    items = getattr(dataset, "items", None)
    if items is None:
        return False
    for item in items:
        routing_mask = getattr(item, "routing_mask", None)
        if routing_mask is not None and not bool(routing_mask.all()):
            return True
    return False


def validate_training_sampler_compatibility(
    *,
    datasets: Sequence[Dataset],
    cost_loss_weight: float,
    sampler_mode: str,
    cost_loss_warmup_epochs: int = 0,
    num_epochs: int | None = None,
) -> None:
    if sampler_mode not in {"standard", "grouped", "after_warmup"}:
        raise ValueError(
            f"unsupported routing group sampler mode: {sampler_mode}"
        )
    cost_loss_can_activate = (
        cost_loss_weight > 0
        and (
            num_epochs is None
            or num_epochs > cost_loss_warmup_epochs
        )
    )
    if not cost_loss_can_activate or sampler_mode != "standard":
        return
    if any(_dataset_has_sparse_routing_views(dataset) for dataset in datasets):
        raise ValueError(
            "standard routing-group sampler mode cannot compute active cost "
            "loss from sparse routing views; use grouped or after_warmup"
        )


def should_build_grouped_training_loader(
    *,
    sampler_mode: str,
    num_epochs: int,
    warmup_epochs: int,
) -> bool:
    if sampler_mode == "grouped":
        return True
    if sampler_mode == "after_warmup":
        return num_epochs > warmup_epochs
    if sampler_mode == "standard":
        return False
    raise ValueError(f"unsupported routing group sampler mode: {sampler_mode}")


def checkpoint_epoch(path: str | Path) -> int | None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        return None
    epoch = checkpoint.get("epoch")
    return int(epoch) if epoch is not None else None


def existing_best_accuracy_checkpoint_path(
    *,
    best_accuracy_model_path: Path,
    legacy_best_model_path: Path,
    best_metric: float,
) -> Path | None:
    if best_metric == float("-inf"):
        return None
    if best_accuracy_model_path.exists():
        return best_accuracy_model_path
    if legacy_best_model_path.exists():
        return legacy_best_model_path
    return None


def is_better_checkpoint(
    metrics: Mapping[str, Any],
    metric_name: str,
    best_metric: float,
    best_cost_metric: float,
    use_cost_tiebreak: bool,
) -> bool:
    candidate_metric = float(metrics[metric_name])
    if candidate_metric > best_metric and not math.isclose(
        candidate_metric,
        best_metric,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return True
    if not use_cost_tiebreak or not math.isclose(
        candidate_metric,
        best_metric,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    candidate_cost_metric = float(metrics["mean_correct_cost_regret"])
    return candidate_cost_metric < best_cost_metric and not math.isclose(
        candidate_cost_metric,
        best_cost_metric,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_epochs < 0:
        raise ValueError("--num-epochs must be non-negative")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.train_eval_every < 0:
        raise ValueError("--train-eval-every must be non-negative")
    if not math.isfinite(args.cost_loss_weight) or args.cost_loss_weight < 0:
        raise ValueError("--cost-loss-weight must be non-negative and finite")
    if not math.isfinite(args.cost_ranking_margin) or args.cost_ranking_margin < 0:
        raise ValueError("--cost-ranking-margin must be non-negative and finite")
    if args.cost_loss_warmup_epochs < 0:
        raise ValueError("--cost-loss-warmup-epochs must be non-negative")
    if args.max_routing_group_views <= 0:
        raise ValueError("--max-routing-group-views must be positive")
    if args.cost_loss_weight > 0:
        if args.cost_source == "none":
            raise ValueError("--cost-loss-weight > 0 cannot use --cost-source none")
        if args.models and not args.require_cost:
            raise ValueError(
                "direct SWE-agent cost-aware training requires --require-cost"
            )
        if args.fill_missing_zero:
            raise ValueError(
                "--fill-missing-zero cannot be used with cost-aware training"
            )
    if args.models and len(set(args.models)) != len(args.models):
        raise ValueError("--models must not contain duplicate model slugs")
    if not math.isfinite(args.embedder_aggregator_dropout) or not (
        0 <= args.embedder_aggregator_dropout < 1
    ):
        raise ValueError("--embedder-aggregator-dropout must be in [0, 1)")
    if not 0 < args.split_ratio < 1:
        raise ValueError("--split-ratio must be between 0 and 1")
    if args.build_embedding_cache_only and args.embedding_cache_path is None:
        raise ValueError("--build-embedding-cache-only requires --embedding-cache-path")
    if args.build_embedding_cache_only and args.shape_probe_only:
        raise ValueError("--build-embedding-cache-only cannot be combined with --shape-probe-only")
    if args.resume_from and args.model_load_path:
        raise ValueError("--resume-from and --model-load-path are mutually exclusive")
    validate_swe_bench_embedding_expansion(args)
    validate_output_paths(args)

    set_seed(args.seed)
    train_generator = torch.Generator().manual_seed(args.seed)
    device = resolve_device(args.device)
    train_dataset, test_dataset, num_models, model_names = build_datasets_from_args(args)
    if args.cost_loss_weight > 0:
        if num_models < 2:
            raise ValueError("cost-aware training requires at least two models")
        validate_cost_aware_datasets((train_dataset, test_dataset))
    validate_training_sampler_compatibility(
        datasets=(train_dataset,),
        cost_loss_weight=args.cost_loss_weight,
        sampler_mode=args.routing_group_sampler_mode,
        cost_loss_warmup_epochs=args.cost_loss_warmup_epochs,
        num_epochs=args.num_epochs,
    )

    if args.build_embedding_cache_only:
        cache_builder_args = argparse.Namespace(
            **{
                **vars(args),
                "embedder_aggregator_n_layers": 0,
            }
        )
        cache_builder = build_router(
            cache_builder_args,
            num_models=num_models,
            device=device,
            use_embedding_cache=False,
        )
        precompute_prompt_chunk_embeddings(
            router=cache_builder,
            datasets=(train_dataset, test_dataset),
            cache_path=args.embedding_cache_path,
            args=args,
        )
        return 0

    standard_train_loader = build_data_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=train_generator,
        preserve_routing_groups=False,
        max_routing_group_views=args.max_routing_group_views,
    )
    grouped_train_loader = (
        build_data_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            generator=train_generator,
            preserve_routing_groups=True,
            max_routing_group_views=args.max_routing_group_views,
        )
        if should_build_grouped_training_loader(
            sampler_mode=args.routing_group_sampler_mode,
            num_epochs=args.num_epochs,
            warmup_epochs=args.cost_loss_warmup_epochs,
        )
        else standard_train_loader
    )
    train_eval_loader = (
        build_data_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if args.train_eval_every > 0
        else None
    )
    test_loader = build_data_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    router = build_router(
        args,
        num_models=num_models,
        device=device,
        use_embedding_cache=not args.shape_probe_only,
    )
    if args.model_load_path:
        load_state_dict_flexibly(router, args.model_load_path, device)

    if args.shape_probe_only:
        run_shape_probe(
            router=router,
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_models=num_models,
        )
        return 0

    validate_router_embedding_cache(
        router,
        datasets=(train_dataset, test_dataset),
    )

    task_aux_head = None
    optimizer_parameters = [parameter for parameter in router.parameters() if parameter.requires_grad]
    if args.task_aux_weight > 0:
        if args.task_aux_classes < 2:
            raise ValueError("--task-aux-classes must be at least 2 when --task-aux-weight > 0")
        task_aux_head = torch.nn.Linear(router.text_dim, args.task_aux_classes).to(device)
        optimizer_parameters += list(task_aux_head.parameters())
    decision_lambda_sampler = (
        LambdaSampler(tuple(args.decision_lambdas), seed=args.seed)
        if args.decision_loss_weight > 0
        else None
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model_ids = torch.arange(num_models, dtype=torch.long, device=device)
    best_model_path = (
        Path(args.best_model_save_path)
        if args.best_model_save_path
        else default_best_model_path(args.model_save_path)
    )
    best_accuracy_model_path = (
        Path(args.best_accuracy_model_save_path)
        if args.best_accuracy_model_save_path
        else default_best_accuracy_model_path(args.model_save_path)
    )
    best_cost_aware_model_path = (
        Path(args.best_cost_aware_model_save_path)
        if args.best_cost_aware_model_save_path
        else default_best_cost_aware_model_path(args.model_save_path)
    )
    last_model_path = (
        Path(args.last_model_save_path)
        if args.last_model_save_path
        else default_last_model_path(args.model_save_path)
    )
    best_metric = float("-inf")
    best_cost_metric = float("inf")
    best_accuracy_epoch: int | None = None
    best_cost_aware_metric = float("-inf")
    best_cost_aware_epoch: int | None = None
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if args.resume_from:
        (
            start_epoch,
            history,
            best_metric,
            best_cost_metric,
            checkpoint_selection_state,
        ) = restore_training_checkpoint(
            path=args.resume_from,
            router=router,
            optimizer=optimizer,
            args=args,
            train_generator=train_generator,
            device=device,
            return_best_cost_metric=True,
            return_checkpoint_selection_state=True,
        )
        best_metric = float(
            checkpoint_selection_state.get(
                "best_accuracy_metric",
                best_metric,
            )
        )
        best_accuracy_epoch_value = checkpoint_selection_state.get(
            "best_accuracy_epoch"
        )
        best_accuracy_epoch = (
            int(best_accuracy_epoch_value)
            if best_accuracy_epoch_value is not None
            else None
        )
        best_cost_aware_metric = float(
            checkpoint_selection_state.get(
                "best_cost_aware_metric",
                float("-inf"),
            )
        )
        best_cost_metric = float(
            checkpoint_selection_state.get(
                "best_cost_aware_cost_metric",
                best_cost_metric,
            )
        )
        best_cost_aware_epoch_value = checkpoint_selection_state.get(
            "best_cost_aware_epoch"
        )
        best_cost_aware_epoch = (
            int(best_cost_aware_epoch_value)
            if best_cost_aware_epoch_value is not None
            else None
        )
        if start_epoch > args.num_epochs + 1:
            raise ValueError(
                f"resume checkpoint completed epoch {start_epoch - 1}, "
                f"which exceeds --num-epochs={args.num_epochs}"
            )
        if (
            best_metric != float("-inf")
            and not best_accuracy_model_path.exists()
            and not best_model_path.exists()
        ):
            raise FileNotFoundError(
                "resume checkpoint references a best-accuracy metric, but "
                f"neither checkpoint exists: {best_accuracy_model_path}, "
                f"{best_model_path}"
            )
        if best_metric != float("-inf") and best_accuracy_epoch is None:
            existing_best_accuracy_path = (
                best_accuracy_model_path
                if best_accuracy_model_path.exists()
                else best_model_path
            )
            best_accuracy_epoch = checkpoint_epoch(
                existing_best_accuracy_path
            )
        if (
            best_cost_aware_metric != float("-inf")
            and not best_cost_aware_model_path.exists()
        ):
            raise FileNotFoundError(
                "resume checkpoint references a best cost-aware metric, but "
                f"the checkpoint is missing: {best_cost_aware_model_path}"
            )
        print(f"resumed training from {args.resume_from} at epoch {start_epoch}")

    metrics_output = args.metrics_output or str(Path(args.model_save_path).with_suffix(".metrics.json"))

    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_cost_loss_weight = effective_cost_loss_weight(
            configured_weight=args.cost_loss_weight,
            warmup_epochs=args.cost_loss_warmup_epochs,
            epoch=epoch,
        )
        train_loader = select_training_loader(
            standard_loader=standard_train_loader,
            grouped_loader=grouped_train_loader,
            sampler_mode=args.routing_group_sampler_mode,
            epoch=epoch,
            warmup_epochs=args.cost_loss_warmup_epochs,
        )
        train_loader_mode = (
            "grouped"
            if isinstance(
                train_loader.batch_sampler,
                RoutingGroupBatchSampler,
            )
            else "standard"
        )
        train_losses = train_one_epoch(
            router=router,
            data_loader=train_loader,
            optimizer=optimizer,
            model_ids=model_ids,
            device=device,
            gradient_clip_norm=args.gradient_clip_norm,
            cost_loss_weight=epoch_cost_loss_weight,
            cost_ranking_margin=args.cost_ranking_margin,
            selection_loss_weight=args.selection_loss_weight,
            selection_temperature=args.selection_temperature,
            decision_loss_weight=args.decision_loss_weight,
            decision_lambda_sampler=decision_lambda_sampler,
            task_aux_head=task_aux_head,
            task_aux_weight=args.task_aux_weight,
            task_label_map=TASK_AUX_LABEL_MAP,
            return_metrics=True,
        )
        if not isinstance(train_losses, TrainingLossMetrics):
            raise TypeError("train_one_epoch did not return loss metrics")

        metrics: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_losses.total_loss,
            "train_accuracy_loss": train_losses.accuracy_loss,
            "train_cost_rank_loss": train_losses.cost_rank_loss,
            "train_supervised_pair_count": train_losses.supervised_pair_count,
            "train_cost_rank_group_count": train_losses.cost_rank_group_count,
            "train_cost_rank_pair_count": train_losses.cost_rank_pair_count,
            "train_selection_loss": train_losses.selection_loss,
            "train_selection_view_count": train_losses.selection_view_count,
            "train_decision_loss": train_losses.decision_loss,
            "train_decision_view_count": train_losses.decision_view_count,
            "train_decision_lambda": train_losses.decision_lambda,
            "train_task_aux_loss": train_losses.task_aux_loss,
            "train_task_aux_view_count": train_losses.task_aux_view_count,
            "effective_cost_loss_weight": epoch_cost_loss_weight,
            "cost_loss_active": epoch_cost_loss_weight > 0,
            "train_loader_mode": train_loader_mode,
        }
        ran_train_eval = should_run_train_eval(epoch, args.train_eval_every)
        if ran_train_eval:
            if train_eval_loader is None:
                raise RuntimeError("train_eval_loader was not initialized")
            train_metrics = evaluate_router(
                router=router,
                data_loader=train_eval_loader,
                model_ids=model_ids,
                device=device,
                model_names=model_names,
            )
            metrics.update(prefix_evaluation_metrics(train_metrics, "train"))

        ran_val_eval = epoch % args.eval_every == 0
        if ran_val_eval:
            val_metrics = evaluate_router(
                router=router,
                data_loader=test_loader,
                model_ids=model_ids,
                device=device,
                model_names=model_names,
            )
            metrics.update(val_metrics)
            if resolve_swe_step_supervision_mode(args) == "all_steps_soft":
                metrics.update(
                    evaluate_step_supervision(
                        router=router,
                        data_loader=test_loader,
                        model_ids=model_ids,
                        device=device,
                    )
                )

        if ran_train_eval or ran_val_eval:
            history.append(dict(metrics))
            print(format_epoch_metrics(metrics))

        if ran_val_eval:
            selected_metric = float(metrics[args.save_best_metric])
            if is_better_checkpoint(
                metrics=metrics,
                metric_name=args.save_best_metric,
                best_metric=best_metric,
                best_cost_metric=float("inf"),
                use_cost_tiebreak=False,
            ):
                best_metric = selected_metric
                best_accuracy_epoch = epoch
                save_checkpoint(
                    best_accuracy_model_path,
                    router,
                    args,
                    epoch,
                    metrics,
                )
                save_checkpoint(best_model_path, router, args, epoch, metrics)
                print(
                    "saved best-accuracy checkpoints to "
                    f"{best_accuracy_model_path} and {best_model_path}"
                )

            cost_aware_checkpoint_eligible = (
                epoch_cost_loss_weight > 0
                and train_losses.cost_rank_pair_count > 0
                and int(metrics.get("correct_cost_regret_prompts", 0)) > 0
            )
            if cost_aware_checkpoint_eligible and is_better_checkpoint(
                metrics=metrics,
                metric_name=args.save_best_metric,
                best_metric=best_cost_aware_metric,
                best_cost_metric=best_cost_metric,
                use_cost_tiebreak=True,
            ):
                best_cost_aware_metric = selected_metric
                best_cost_metric = float(
                    metrics["mean_correct_cost_regret"]
                )
                best_cost_aware_epoch = epoch
                save_checkpoint(
                    best_cost_aware_model_path,
                    router,
                    args,
                    epoch,
                    metrics,
                )
                print(
                    "saved best cost-aware checkpoint to "
                    f"{best_cost_aware_model_path}"
                )

        save_training_checkpoint(
            path=last_model_path,
            router=router,
            optimizer=optimizer,
            args=args,
            completed_epoch=epoch,
            metrics=metrics,
            history=history,
            best_metric=best_metric,
            best_cost_metric=best_cost_metric,
            checkpoint_selection_state={
                "best_accuracy_metric": best_metric,
                "best_accuracy_epoch": best_accuracy_epoch,
                "best_cost_aware_metric": best_cost_aware_metric,
                "best_cost_aware_cost_metric": best_cost_metric,
                "best_cost_aware_epoch": best_cost_aware_epoch,
            },
            train_generator=train_generator,
        )
        print(f"saved last checkpoint to {last_model_path}")
        current_best_accuracy_path = existing_best_accuracy_checkpoint_path(
            best_accuracy_model_path=best_accuracy_model_path,
            legacy_best_model_path=best_model_path,
            best_metric=best_metric,
        )
        write_json(
            metrics_output,
            {
                "args": _json_safe(vars(args)),
                "num_models": num_models,
                "model_names": model_names,
                "best_metric": best_metric if best_metric != float("-inf") else None,
                "best_accuracy_metric": (
                    best_metric if best_metric != float("-inf") else None
                ),
                "best_accuracy_epoch": best_accuracy_epoch,
                "best_accuracy_checkpoint": (
                    str(current_best_accuracy_path)
                    if current_best_accuracy_path is not None
                    else None
                ),
                "legacy_best_checkpoint": (
                    str(best_model_path)
                    if best_metric != float("-inf") and best_model_path.exists()
                    else None
                ),
                "best_cost_aware_metric": (
                    best_cost_aware_metric
                    if best_cost_aware_metric != float("-inf")
                    else None
                ),
                "best_cost_metric": (
                    best_cost_metric if best_cost_metric != float("inf") else None
                ),
                "best_cost_aware_epoch": best_cost_aware_epoch,
                "best_cost_aware_checkpoint": (
                    str(best_cost_aware_model_path)
                    if best_cost_aware_epoch is not None
                    else None
                ),
                "final_metrics": None,
                "history": history,
            },
        )

    final_metrics = evaluate_router(
        router=router,
        data_loader=test_loader,
        model_ids=model_ids,
        device=device,
        model_names=model_names,
    )
    if resolve_swe_step_supervision_mode(args) == "all_steps_soft":
        final_metrics.update(
            evaluate_step_supervision(
                router=router,
                data_loader=test_loader,
                model_ids=model_ids,
                device=device,
            )
        )
    save_checkpoint(args.model_save_path, router, args, args.num_epochs, final_metrics)
    print(f"saved final checkpoint to {args.model_save_path}")

    current_best_accuracy_path = existing_best_accuracy_checkpoint_path(
        best_accuracy_model_path=best_accuracy_model_path,
        legacy_best_model_path=best_model_path,
        best_metric=best_metric,
    )
    write_json(
        metrics_output,
        {
            "args": _json_safe(vars(args)),
            "num_models": num_models,
            "model_names": model_names,
            "best_metric": best_metric if best_metric != float("-inf") else None,
            "best_accuracy_metric": (
                best_metric if best_metric != float("-inf") else None
            ),
            "best_accuracy_epoch": best_accuracy_epoch,
            "best_accuracy_checkpoint": (
                str(current_best_accuracy_path)
                if current_best_accuracy_path is not None
                else None
            ),
            "legacy_best_checkpoint": (
                str(best_model_path)
                if best_metric != float("-inf") and best_model_path.exists()
                else None
            ),
            "best_cost_aware_metric": (
                best_cost_aware_metric
                if best_cost_aware_metric != float("-inf")
                else None
            ),
            "best_cost_metric": (
                best_cost_metric if best_cost_metric != float("inf") else None
            ),
            "best_cost_aware_epoch": best_cost_aware_epoch,
            "best_cost_aware_checkpoint": (
                str(best_cost_aware_model_path)
                if best_cost_aware_epoch is not None
                else None
            ),
            "final_metrics": final_metrics,
            "history": history,
        },
    )
    print(f"wrote metrics to {metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
