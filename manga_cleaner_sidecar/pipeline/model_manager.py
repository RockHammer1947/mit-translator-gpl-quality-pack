from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Any

from manga_cleaner_sidecar.contracts import CleanerError

LAMA_LARGE_MODEL_URL = "https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt"
LAMA_LARGE_MODEL_SHA256 = "11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935"
LAMA_LARGE_MODEL_NAME = "lama_large_512px.ckpt"


def default_model_dir() -> Path:
    configured = os.environ.get("MANGA_CLEANER_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "manga-cleaner-sidecar" / "models" / "lama"


def resolve_lama_large_model_path(model_path: Path | None = None) -> Path:
    configured = model_path or (Path(os.environ["MANGA_CLEANER_LAMA_MODEL"]).expanduser() if os.environ.get("MANGA_CLEANER_LAMA_MODEL") else None)
    return configured if configured is not None else default_model_dir() / LAMA_LARGE_MODEL_NAME


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_lama_large_model(model_path: Path | None = None) -> dict[str, Any]:
    resolved = resolve_lama_large_model_path(model_path)
    exists = resolved.exists()
    digest = sha256_file(resolved) if exists and resolved.is_file() else None
    return {
        "path": str(resolved),
        "name": resolved.name,
        "exists": exists,
        "sha256": digest,
        "expected_sha256": LAMA_LARGE_MODEL_SHA256,
        "hash_ok": bool(digest == LAMA_LARGE_MODEL_SHA256) if digest else False,
        "url": LAMA_LARGE_MODEL_URL,
    }


def prepare_lama_large_model(model_path: Path | None = None, *, force: bool = False) -> dict[str, Any]:
    resolved = resolve_lama_large_model_path(model_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not force:
        info = inspect_lama_large_model(resolved)
        if info["hash_ok"]:
            return {**info, "downloaded": False, "status": "ready"}
    tmp = resolved.with_suffix(resolved.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(LAMA_LARGE_MODEL_URL, tmp)
        tmp.replace(resolved)
    except Exception as error:  # pragma: no cover - network failures are environment-specific.
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise CleanerError("MODEL_DOWNLOAD_FAILED", f"Failed to download LaMa model: {error}") from error
    info = inspect_lama_large_model(resolved)
    if not info["hash_ok"]:
        raise CleanerError("MODEL_HASH_MISMATCH", f"LaMa model hash mismatch at {resolved}")
    return {**info, "downloaded": True, "status": "ready"}

