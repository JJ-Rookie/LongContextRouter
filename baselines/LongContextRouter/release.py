"""Train, fit a task head, evaluate, or route with the fixed five-model v7 recipe."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import causal_cost, release_io as io, task_router, train
from .training_io import dataset_fingerprint

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/historical.json"


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "fit-task-head", "evaluate", "route"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--cache", type=Path, required=True)
        if name == "train":
            command.add_argument("--train-csv", type=Path, required=True)
            command.add_argument("--validation-csv", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--device", default="cpu")
            continue
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=32)
        if name == "fit-task-head":
            command.add_argument("--train-rows", type=Path, required=True)
            command.add_argument("--validation-rows", type=Path, required=True)
        else:
            command.add_argument("--task-head", type=Path, required=True)
            command.add_argument("--cost-profile", type=Path, required=True)
            command.add_argument("--rows" if name == "evaluate" else "--inputs", type=Path, required=True)
    return parser


def _training_argv(args, config):
    parser_defaults = vars(train.parse_args([]))
    reserved = {"models", "train_data_path", "test_data_path", "model_save_path", "metrics_output",
                "embedding_cache_path", "device", "resume_from", "model_load_path"}
    settings = config["training"]
    if set(settings) - set(parser_defaults) or set(settings) & reserved:
        raise ValueError("training configuration contains unsupported or reserved options")
    argv = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            argv.extend([flag] if value else [])
        elif isinstance(value, (list, tuple)):
            argv.extend([flag, *map(str, value)])
        elif value is not None:
            argv.extend([flag, str(value)])
    return argv + ["--train-data-path", str(args.train_csv), "--test-data-path", str(args.validation_csv),
                   "--model-num", str(len(io.MODELS)), "--embedding-cache-path", str(args.cache),
                   "--device", args.device, "--model-save-path", str(args.output_dir / "saved_model.pth"),
                   "--metrics-output", str(args.output_dir / "training_metrics.json")]


def validate_training_splits(training, validation, config):
    groups = [{item.routing_group_id for item in dataset.items} for dataset in (training, validation)]
    if not all(groups) or groups[0] & groups[1]:
        raise ValueError("Training and validation routing groups must be nonempty and disjoint")
    for dataset in (training, validation):
        if tuple(dataset.model_names.get(index) for index in range(5)) != io.MODELS:
            raise ValueError("Prepared CSV model order does not match v7")
        if any(item.category.split("/", 1)[0] not in io.TASKS for item in dataset.items):
            raise ValueError("Prepared CSV contains an excluded benchmark")
    expected = config["training"].get("expected_dataset_fingerprint")
    if expected and dataset_fingerprint((training, validation)) != expected:
        raise ValueError("Prepared CSV does not reproduce the configured training fingerprint")


def _train(args, config):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse nonempty training directory {args.output_dir}")
    argv = _training_argv(args, config)
    training, validation, count, names = train.build_csv_datasets(train.parse_args(argv))
    if count != len(io.MODELS) or tuple(names.get(index) for index in range(count)) != io.MODELS:
        raise ValueError("Prepared CSV does not match the five-model pool")
    validate_training_splits(training, validation, config)
    io.validate_training_cache(args.cache, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = train.main(argv)
    if status == 0:
        io.write_json(args.output_dir / "release_manifest.json", {
            "models": list(io.MODELS), "tasks": list(io.TASKS), "training": config["training"],
            "checkpoints": {path.name: io.sha256(path) for path in sorted(args.output_dir.glob("*.pth"))}})
    return status


def _fit(args, config):
    output = io.new_output(args.output)
    training, validation = io.read_rows(args.train_rows, "fit"), io.read_rows(args.validation_rows, "fit")
    if {row["prompt_id"] for row in training} & {row["prompt_id"] for row in validation}:
        raise ValueError("Task-head training and validation prompt IDs must be disjoint")
    runtime = io.load_runtime(args.checkpoint, args.cache, config)
    train_x, _ = runtime.outputs([row["prompt"] for row in training], args.batch_size)
    val_x, _ = runtime.outputs([row["prompt"] for row in validation], args.batch_size)
    train_y = torch.tensor([io.TASKS.index(row["task"]) for row in training])
    val_y = torch.tensor([io.TASKS.index(row["task"]) for row in validation])
    fitted = task_router.fit_task_head(train_x, train_y, val_x, val_y, io.TASKS, **config["task_head"])
    with output.open("xb") as handle:
        torch.save({**fitted, "models": list(io.MODELS),
                    "router_checkpoint_sha256": io.sha256(args.checkpoint)}, handle)
    return 0


def _selection(config, profile):
    policy = config["policy"]
    guard = policy["guard"]
    return {
        "tasks": io.TASKS, "task_profile": profile,
        "task_policy": {task: {"weight": policy["weights"][task],
            "anchor": policy.get("anchors", {}).get(task) if guard else None,
            "threshold": policy.get("thresholds", {}).get(task) if guard else None,
            "anchor_only": policy.get("anchor_only", {}).get(task, False) if guard else False}
            for task in io.TASKS},
        "mode": policy["mode"], "confidence_threshold": policy["threshold"],
        "fallback_weight": policy["fallback"], "fallback_scale": "train_log1p_median",
        "cost_scale": policy["scale"],
        "cost_scale_tasks": () if policy["scale"] == "minmax" else tuple(policy["scope"]),
        "guard_enabled": guard,
    }


def _predict(args, config):
    io.new_output(args.output)
    rows = io.read_rows(args.rows if args.command == "evaluate" else args.inputs, args.command)
    profile = causal_cost.load_profile(args.cost_profile, io.MODELS)
    expected = config.get("artifacts", {}).get("cost_profile_sha256")
    if expected and io.sha256(args.cost_profile) != expected:
        raise ValueError("Cost profile differs from the configured historical profile SHA256")
    runtime = io.load_runtime(args.checkpoint, args.cache, config)
    features, scores = runtime.outputs([row["prompt"] for row in rows], args.batch_size)
    checkpoint_sha256 = io.sha256(args.checkpoint)
    head = io.load_task_head(args.task_head, config, checkpoint_sha256)
    posteriors = task_router.predict_probabilities(head, features).tolist()
    selection = _selection(config, profile)
    if args.command == "evaluate":
        records = [{**row, "scores": score} for row, score in zip(rows, scores.tolist())]
        result = task_router.evaluate_records(records, models=io.MODELS,
            task_posteriors={row["prompt_id"]: posterior for row, posterior in zip(rows, posteriors)},
            **selection)
    else:
        decisions = [task_router.route_decision(scores=score, task_posterior=posterior,
                     models=io.MODELS, **selection) for score, posterior in zip(scores.tolist(), posteriors)]
        result = {"models": list(io.MODELS), "decisions": [
            {"prompt_id": row["prompt_id"], **decision,
             "selected_model": io.MODELS[decision["selected_model_id"]]}
            for row, decision in zip(rows, decisions)]}
    cache = io.ChunkEmbeddingCache.load(args.cache)
    provenance = {"version": config.get("version"), "models": list(io.MODELS),
                  "checkpoint_sha256": checkpoint_sha256,
                  "cache_fingerprint": cache.fingerprint,
                  "source_cache_fingerprint": cache.config.get("source_cache_fingerprint"),
                  "cost_profile_sha256": io.sha256(args.cost_profile),
                  "task_head_sha256": io.sha256(args.task_head), "policy": config["policy"],
                  "protocol": config.get("protocol")}
    io.write_json(args.output, {**result, "provenance": provenance})
    return 0


def main(argv=None):
    args = _parser().parse_args(argv)
    config = io.load_config(args.config)
    if args.command == "train":
        return _train(args, config)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.command == "fit-task-head":
        return _fit(args, config)
    return _predict(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
