"""Validated, portable artifact I/O for the five-model v7 release."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PosixPath

import torch

from . import task_router, train
from .embedding_cache import ChunkEmbeddingCache

MODELS = ("deepseek-v4-pro", "deepseek-v4-flash-0731", "deepseek-v4-pro-0813",
          "qwen3.5-122b-a10b", "glm-4-flash-250414")
TASKS = ("bbh", "memory_agent_bench", "mmau", "swe_bench_pro")
ENCODER = "Qwen/Qwen3-Embedding-0.6B"


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_config(path):
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    if tuple(config.get("models", ())) != MODELS or tuple(config.get("tasks", ())) != TASKS:
        raise ValueError("Configuration must preserve the v7 five-model and four-task order")
    if not all(isinstance(config.get(key), dict) for key in ("training", "task_head", "policy")):
        raise ValueError("Configuration requires training, task_head and policy objects")
    if set(config["policy"].get("weights", {})) != set(TASKS):
        raise ValueError("Policy requires one weight for each task")
    return config


def _validate_row(row, kind):
    if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                       for k in ("prompt_id", "prompt")):
        raise ValueError("Each row requires nonempty prompt_id and prompt strings")
    if kind == "route" and set(row) != {"prompt_id", "prompt"}:
        raise ValueError("Routing inputs accept only prompt_id and prompt; outcomes are excluded")
    if kind == "fit" and row.get("task") not in TASKS:
        raise ValueError("Task-head training rows require a known task")
    if kind != "evaluate":
        return
    category = row.get("category")
    if not isinstance(category, str) or category.split("/", 1)[0] not in TASKS:
        raise ValueError("Evaluation rows require a known category")
    labels, costs = row.get("labels"), row.get("costs")
    if (not isinstance(labels, list) or len(labels) != len(MODELS)
            or any(type(value) is not int or value not in (0, 1) for value in labels)):
        raise ValueError("Evaluation labels must contain five binary integers in model order")
    if (not isinstance(costs, list) or len(costs) != len(MODELS)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in costs)):
        raise ValueError("Evaluation costs must contain five finite nonnegative numbers")


def read_rows(path, kind):
    if kind not in {"route", "fit", "evaluate"}:
        raise ValueError("Unknown input row kind")
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line, parse_constant=_reject_constant) for line in handle if line.strip()]
    for row in rows:
        _validate_row(row, kind)
    identifiers = [row["prompt_id"] for row in rows]
    if not rows or len(identifiers) != len(set(identifiers)):
        raise ValueError("Input rows must be nonempty and have unique prompt IDs")
    return rows


def new_output(path):
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path, payload):
    output = new_output(path)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def load_checkpoint(path):
    with torch.serialization.safe_globals([PosixPath]):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or not isinstance(payload.get("args"), dict)
            or not isinstance(payload.get("model_state_dict"), dict)):
        raise ValueError("Checkpoint requires args and model_state_dict mappings")
    return payload


def _checkpoint_models(path, payload):
    saved = payload["args"]
    if saved.get("models") is not None:
        return tuple(saved["models"])
    manifest = read_json(Path(path).with_name("release_manifest.json"))
    if manifest.get("checkpoints", {}).get(Path(path).name) != sha256(path):
        raise ValueError("CSV-trained checkpoint must match its release_manifest.json SHA256")
    return tuple(manifest.get("models", ()))


def _cache_arguments(saved, cache):
    metadata = cache.config
    if metadata.get("model_name") != ENCODER:
        raise ValueError("Convert the cache with portable_cache before using this entrypoint")
    source_name = metadata.get("source_model_name", ENCODER)
    if saved.get("embedder_model_name") not in {ENCODER, source_name}:
        raise ValueError("Checkpoint encoder does not match the cache source encoder")
    expected = {"max_chunk_length": int(saved.get("embedder_max_chunk_length_k", 0) * 1024),
                "overlap_tokens": int(saved.get("embedder_overlap_tokens_k", 0) * 1024),
                "backend": saved.get("embedder_backend"), "embedding_dim": 1024}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint and cache chunk settings, backend or dimension disagree")
    if expected != {"max_chunk_length": 32768, "overlap_tokens": 512,
                    "backend": "vllm", "embedding_dim": 1024}:
        raise ValueError("Cache does not use the historical v7 embedding recipe")
    return {**saved, "embedder_model_name": ENCODER}


def _validate_cache_origin(cache, config):
    artifacts = config.get("artifacts", {})
    expected_encoder = artifacts.get("encoder_source_fingerprint")
    if expected_encoder and cache.config.get("source_model_source_fingerprint") != expected_encoder:
        raise ValueError("Cache was not produced by the historical encoder source fingerprint")
    allowed = {artifacts[key] for key in ("training_cache_fingerprint", "query_cache_fingerprint",
                                         "test_cache_fingerprint") if artifacts.get(key)}
    if allowed and cache.config.get("source_cache_fingerprint") not in allowed:
        raise ValueError("Cache source fingerprint is not one of the configured historical views")


def _validate_architecture(saved, config):
    fields = ("fusion_mode", "embedder_aggregator_n_layers", "embedder_aggregator_dropout",
              "no_embedder_qk_norm", "embedder_rope_base", "alpha")
    for field in fields:
        if field not in config["training"] or saved.get(field) != config["training"][field]:
            raise ValueError(f"Checkpoint architecture does not match configured v7: {field}")


class RouterRuntime:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def outputs(self, prompts, batch_size=32):
        if not prompts or type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Outputs require prompts and a positive integer batch size")
        features_out, scores_out = [], []
        model_ids = torch.arange(len(MODELS), device="cpu")
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            features = self.model.embedder(batch).detach().cpu()
            logits = self.model._classify_prompt_embeddings(model_ids, features, test_mode=True)
            if tuple(logits.shape) != (len(batch), len(MODELS), 2):
                raise ValueError("Router returned incompatible logits")
            scores = (logits[:, :, 1] - logits[:, :, 0]).detach().cpu()
            if not bool(torch.isfinite(features).all() and torch.isfinite(scores).all()):
                raise ValueError("Router returned nonfinite features or scores")
            features_out.append(features)
            scores_out.append(scores)
        return torch.cat(features_out), torch.cat(scores_out)


def load_runtime(checkpoint_path, cache_path, config):
    payload = load_checkpoint(checkpoint_path)
    if _checkpoint_models(checkpoint_path, payload) != tuple(config["models"]):
        raise ValueError("Checkpoint model order does not match the configured v7 pool")
    _validate_architecture(payload["args"], config)
    cache = ChunkEmbeddingCache.load(cache_path)
    _validate_cache_origin(cache, config)
    saved = _cache_arguments(payload["args"], cache)
    defaults = vars(train.parse_args([]))
    args = argparse.Namespace(**{**defaults, **saved, "embedding_cache_path": Path(cache_path)})
    model = train.build_router(args, len(MODELS), torch.device("cpu"), use_embedding_cache=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return RouterRuntime(model)


def validate_training_cache(path, config):
    cache = ChunkEmbeddingCache.load(path)
    _validate_cache_origin(cache, config)
    settings = {**vars(train.parse_args([])), **config["training"]}
    _cache_arguments(settings, cache)
    expected = config.get("artifacts", {}).get("training_cache_fingerprint")
    source = cache.config.get("source_cache_fingerprint", cache.fingerprint)
    if expected and source != expected:
        raise ValueError("Training cache source fingerprint does not match historical v7")
    return cache.fingerprint


def load_task_head(path, config, expected_checkpoint_sha256=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or tuple(payload.get("tasks", ())) != tuple(config["tasks"]):
        raise ValueError("Task-head task order does not match the configured v7 tasks")
    if "models" in payload and tuple(payload["models"]) != tuple(config["models"]):
        raise ValueError("Task-head model order does not match the configured v7 pool")
    if expected_checkpoint_sha256 and payload.get("router_checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("Task head was not fitted on this router checkpoint; run fit-task-head")
    return task_router.task_head_from_payload(payload)
