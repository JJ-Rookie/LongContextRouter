from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from .embedding_cache import CACHE_FORMAT_VERSION, _normalized_config


PORTABLE_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
_EXPECTED_CONFIG = {
    "max_chunk_length": 32768,
    "overlap_tokens": 512,
    "embedding_dim": 1024,
    "backend": "vllm",
}


def _validate_payload(payload) -> dict:
    if not isinstance(payload, dict) or payload.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError("unsupported embedding cache format")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("embedding cache config must be a mapping")
    for key, expected in _EXPECTED_CONFIG.items():
        if type(config.get(key)) is not type(expected) or config[key] != expected:
            raise ValueError(f"historical cache requires {key}={expected!r}")
    model_name = config.get("model_name")
    if not isinstance(model_name, str) or Path(model_name).name != "Qwen3-Embedding-0.6B":
        raise ValueError("source encoder must be Qwen3-Embedding-0.6B")
    hashes, entries = payload.get("prompt_hashes"), payload.get("entries")
    if (not isinstance(hashes, (list, tuple)) or not hashes
            or any(not isinstance(key, str) or not key for key in hashes)):
        raise ValueError("cache prompt keys must be nonempty strings")
    if len(hashes) != len(set(hashes)) or not isinstance(entries, dict) or set(entries) != set(hashes):
        raise ValueError("cache entries do not match unique prompt hashes")
    for tensor in entries.values():
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] != 1024
                or tensor.layout != torch.strided or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError("cache entries must be finite contiguous float32 [chunks, 1024] tensors")
    return _normalized_config(config)


def _save_exclusive(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def migrate_cache(source: str | Path, output: str | Path) -> dict:
    source_path, output_path = Path(source), Path(output)
    if os.path.lexists(output_path):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    with source_path.open("rb") as handle:
        payload = torch.load(handle, map_location="cpu", weights_only=True)
        config = _validate_payload(payload)
    portable_config = {
        **config,
        "model_name": PORTABLE_MODEL_ID,
        "source_model_name": config.get("source_model_name", config["model_name"]),
    }
    _save_exclusive({
        "format_version": CACHE_FORMAT_VERSION,
        "config": portable_config,
        "prompt_hashes": payload["prompt_hashes"],
        "entries": payload["entries"],
    }, output_path)
    return {"source": str(source_path), "output": str(output_path),
            "config": portable_config}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert embedding cache metadata for portable use.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = migrate_cache(args.source, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
