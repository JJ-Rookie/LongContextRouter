"""Rebind a verified historical embedding cache without encoding any text.

The original model identity, cache fingerprint, and file hash remain inside the
new fingerprinted config. This verifies cache integrity, not encoder weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from .embedding_cache import CACHE_FORMAT_VERSION, _build_cache_fingerprint_from_hashes


PORTABLE_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
_EXPECTED_CONFIG = {
    "max_chunk_length": 32768,
    "overlap_tokens": 512,
    "embedding_dim": 1024,
    "backend": "vllm",
}
_PROVENANCE_KEYS = (
    "source_model_name", "source_model_source_fingerprint",
    "source_cache_fingerprint", "source_file_sha256",
)


def _sha256(handle) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _is_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


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
    if not _is_digest(config.get("model_source_fingerprint")):
        raise ValueError("source model fingerprint must be a SHA-256 digest")
    if any(key in config for key in _PROVENANCE_KEYS):
        raise ValueError("source cache already contains portable provenance")
    hashes, entries = payload.get("prompt_hashes"), payload.get("entries")
    if not isinstance(hashes, (list, tuple)) or not hashes or not all(map(_is_digest, hashes)):
        raise ValueError("cache prompt hashes must be nonempty SHA-256 digests")
    if len(hashes) != len(set(hashes)) or not isinstance(entries, dict) or set(entries) != set(hashes):
        raise ValueError("cache entries do not match unique prompt hashes")
    for tensor in entries.values():
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] != 1024
                or tensor.layout != torch.strided or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError("cache entries must be finite contiguous float32 [chunks, 1024] tensors")
    expected = _build_cache_fingerprint_from_hashes(config, hashes, entries)
    if payload.get("fingerprint") != expected:
        raise ValueError("embedding cache fingerprint does not match its contents")
    return dict(config)


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
        # The temporary file is on the same filesystem; link is atomic and fails
        # if any output entry already exists, including a dangling symlink.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def migrate_cache(source: str | Path, output: str | Path) -> dict:
    """Return output metadata; preserve every prompt hash and float32 tensor."""
    source_path, output_path = Path(source), Path(output)
    if os.path.lexists(output_path):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    with source_path.open("rb") as handle:
        source_sha256 = _sha256(handle)
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
        config = _validate_payload(payload)
        if _sha256(handle) != source_sha256:
            raise ValueError("source cache changed during migration")
    portable_config = {
        **config,
        "model_name": PORTABLE_MODEL_ID,
        "model_source_fingerprint": hashlib.sha256(
            f"model-id:{PORTABLE_MODEL_ID}".encode("utf-8")
        ).hexdigest(),
        "source_model_name": config["model_name"],
        "source_model_source_fingerprint": config["model_source_fingerprint"],
        "source_cache_fingerprint": payload["fingerprint"],
        "source_file_sha256": source_sha256,
    }
    fingerprint = _build_cache_fingerprint_from_hashes(
        portable_config, payload["prompt_hashes"], payload["entries"]
    )
    _save_exclusive({**payload, "config": portable_config, "fingerprint": fingerprint}, output_path)
    return {"source": str(source_path), "output": str(output_path),
            "fingerprint": fingerprint, "config": portable_config}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
