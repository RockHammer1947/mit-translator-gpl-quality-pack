from __future__ import annotations

import os
import platform
import shlex
import shutil
from pathlib import Path
from typing import Any

from manga_cleaner_sidecar import __version__
from manga_cleaner_sidecar.contracts import SCHEMA_VERSION
from manga_cleaner_sidecar.pipeline.model_manager import inspect_lama_large_model


def run_doctor(model_path: Path | None = None, lama_command: str | None = None) -> dict[str, Any]:
    command = lama_command or os.environ.get("MANGA_CLEANER_LAMA_CMD")
    lama_available = bool(command and shutil.which(shlex.split(command)[0]))
    lama_reason = None
    if not command:
        lama_reason = "set --lama-command or MANGA_CLEANER_LAMA_CMD to enable lama-large"
    elif not lama_available:
        lama_reason = "lama-large adapter command was not found"
    elif model_path is not None and not model_path.exists():
        lama_available = False
        lama_reason = "model path was not found"

    model_info = inspect_lama_large_model(model_path)
    try:
        import torch  # type: ignore

        torch_available = True
        torch_version = getattr(torch, "__version__", None)
        mps_available = bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())
    except Exception:
        torch_available = False
        torch_version = None
        mps_available = False

    lama_internal_available = bool(model_info["exists"] and torch_available)
    lama_internal_reason = None
    if not model_info["exists"]:
        lama_internal_reason = "model not found; run prepare-models --provider lama-large-internal"
    elif not torch_available:
        lama_internal_reason = "torch is not installed"

    return {
        "type": "doctor",
        "job_id": "doctor",
        "schema_version": SCHEMA_VERSION,
        "sidecar_version": __version__,
        "python_version": platform.python_version(),
        "providers": [
            {
                "name": "lama-large-internal",
                "available": lama_internal_available,
                "model_ready": bool(model_info["exists"]),
                "model_path": model_info["path"],
                "model_hash_ok": model_info["hash_ok"],
                "torch_available": torch_available,
                "torch_version": torch_version,
                "device": "mps" if mps_available else "cpu",
                "precision": "bf16",
                "inpainting_size": 2048,
                **({"reason": lama_internal_reason} if lama_internal_reason else {}),
            },
            {"name": "telea", "available": True},
            {"name": "none", "available": True},
            {
                "name": "lama-large",
                "available": lama_available,
                **({"reason": lama_reason} if lama_reason else {}),
                **({"command": command} if command else {}),
                **({"model_path": str(model_path)} if model_path else {}),
            },
        ],
    }
