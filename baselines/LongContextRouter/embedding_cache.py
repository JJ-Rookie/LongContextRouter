from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


CACHE_FORMAT_VERSION = 2


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "model_name", "max_chunk_length", "overlap_tokens", "embedding_dim",
        "backend", "source_model_name",
    }
    normalized = {
        str(key): str(value) if isinstance(value, Path) else value
        for key, value in config.items()
        if str(key) in fields
    }
    required = {
        "model_name",
        "max_chunk_length",
        "overlap_tokens",
        "embedding_dim",
    }
    missing = required - set(normalized)
    if missing:
        raise ValueError(f"embedding cache config missing fields: {sorted(missing)}")
    return normalized


def _normalized_entries(
    entries: Mapping[str, torch.Tensor],
    *,
    embedding_dim: int,
) -> dict[str, torch.Tensor]:
    normalized = {
        str(key): clone_chunk_embeddings_for_cache(value)
        for key, value in entries.items()
    }
    for key, tensor in normalized.items():
        if tensor.size(1) != embedding_dim:
            raise ValueError(
                f"cache entry {key} has embedding dim {tensor.size(1)}, "
                f"expected {embedding_dim}"
            )
    return normalized


def clone_chunk_embeddings_for_cache(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(
            "chunk embeddings must be tensors with shape [num_chunks, embedding_dim]"
        )
    if tensor.size(0) <= 0:
        raise ValueError("chunk embeddings must contain at least one chunk")
    with torch.inference_mode(False):
        result = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("chunk embeddings must contain only finite values")
    return result


class ChunkEmbeddingCache:

    def __init__(
        self,
        *,
        entries: Mapping[str, torch.Tensor],
        config: Mapping[str, Any],
        prompts: Sequence[str],
    ) -> None:
        self._config = _normalized_config(config)
        self._prompt_hashes = tuple(sorted({prompt_hash(prompt) for prompt in prompts}))
        self._entries = _normalized_entries(
            entries,
            embedding_dim=int(self._config["embedding_dim"]),
        )
        if set(self._entries) != set(self._prompt_hashes):
            raise ValueError("embedding cache entries do not match prompt hashes")

    @classmethod
    def build(
        cls,
        *,
        prompts: Sequence[str],
        chunk_embeddings_by_prompt: Mapping[str, torch.Tensor],
        config: Mapping[str, Any],
    ) -> "ChunkEmbeddingCache":
        if not prompts:
            raise ValueError("cannot build an embedding cache without prompts")
        missing = [prompt for prompt in prompts if prompt not in chunk_embeddings_by_prompt]
        if missing:
            raise ValueError(f"missing chunk embeddings for {len(missing)} prompts")
        entries: dict[str, torch.Tensor] = {}
        for prompt in prompts:
            key = prompt_hash(prompt)
            candidate = clone_chunk_embeddings_for_cache(chunk_embeddings_by_prompt[prompt])
            previous = entries.get(key)
            if previous is not None and not torch.equal(previous, candidate):
                raise ValueError("prompt hash collision produced different chunk embeddings")
            entries[key] = candidate
        normalized_config = _normalized_config(config)
        return cls(
            entries=entries,
            config=normalized_config,
            prompts=prompts,
        )

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def validate_prompts(self, prompts: Sequence[str]) -> None:
        expected_hashes = tuple(sorted({prompt_hash(prompt) for prompt in prompts}))
        if expected_hashes != self._prompt_hashes:
            raise ValueError("embedding cache keys do not match prompts")

    def get_batch(
        self,
        prompts: Sequence[str],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if not prompts:
            embedding_dim = int(self._config["embedding_dim"])
            target = torch.device(device)
            return (
                torch.empty(0, 0, embedding_dim, dtype=torch.float32, device=target),
                torch.empty(0, 0, 0, dtype=torch.bool, device=target),
                [],
            )

        values: list[torch.Tensor] = []
        missing: list[str] = []
        for prompt in prompts:
            value = self._entries.get(prompt_hash(prompt))
            if value is None:
                missing.append(prompt_hash(prompt)[:12])
            else:
                values.append(value)
        if missing:
            raise ValueError(f"embedding cache is missing prompts: {missing}")

        target = torch.device(device)
        chunk_counts = [int(value.size(0)) for value in values]
        padded = torch.nn.utils.rnn.pad_sequence(values, batch_first=True).to(target)
        max_chunks = padded.size(1)
        indices = torch.arange(max_chunks, device=target)
        causal = indices.unsqueeze(0) > indices.unsqueeze(1)
        counts = torch.tensor(chunk_counts, device=target)
        padding = indices.unsqueeze(0) >= counts.unsqueeze(1)
        attention_mask = causal.unsqueeze(0) | padding.unsqueeze(2) | padding.unsqueeze(1)
        return padded, attention_mask, chunk_counts

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "config": dict(self._config),
            "prompt_hashes": list(self._prompt_hashes),
            "entries": {key: value.clone() for key, value in self._entries.items()},
        }
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_config: Mapping[str, Any] | None = None,
        expected_prompts: Sequence[str] | None = None,
    ) -> "ChunkEmbeddingCache":
        cache_path = Path(path)
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != CACHE_FORMAT_VERSION:
            raise ValueError("unsupported embedding cache format")

        config = _normalized_config(payload.get("config", {}))
        prompt_hashes = tuple(str(value) for value in payload.get("prompt_hashes", []))
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("embedding cache entries are invalid")

        if expected_config is not None and config != _normalized_config(expected_config):
            raise ValueError("embedding cache does not match expected config")
        expected_hashes = (
            tuple(sorted({prompt_hash(prompt) for prompt in expected_prompts}))
            if expected_prompts is not None
            else tuple(sorted(prompt_hashes))
        )
        if expected_hashes != tuple(sorted(prompt_hashes)):
            raise ValueError("embedding cache keys do not match expected prompts")

        normalized_entries = _normalized_entries(
            entries,
            embedding_dim=int(config["embedding_dim"]),
        )
        if set(normalized_entries) != set(expected_hashes):
            raise ValueError("embedding cache entries do not match prompt hashes")
        instance = cls.__new__(cls)
        instance._config = config
        instance._prompt_hashes = expected_hashes
        instance._entries = normalized_entries
        return instance
