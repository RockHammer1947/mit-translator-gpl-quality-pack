from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import typer

from manga_cleaner_sidecar import __version__
from manga_cleaner_sidecar.contracts import CleanerError, SCHEMA_VERSION


def make_job_id(prefix: str = "job_manga_cleaner") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def emit(event: dict[str, Any], jsonl: bool) -> None:
    if not jsonl:
        return
    event.setdefault("schema_version", SCHEMA_VERSION)
    typer.echo(json.dumps(event, ensure_ascii=False, separators=(",", ":")))


def emit_started(job_id: str, provider: str, jsonl: bool) -> None:
    emit(
        {
            "type": "started",
            "job_id": job_id,
            "sidecar_version": __version__,
            "provider": provider,
        },
        jsonl,
    )


def dump_json(path: Path, payload: dict[str, Any], *, manifest: bool = False) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as error:
        code = "MANIFEST_WRITE_FAILED" if manifest else "OUTPUT_WRITE_FAILED"
        raise CleanerError(code, f"Failed to write {path}: {error}") from error


def print_error(error: Exception, job_id: str, jsonl: bool) -> None:
    code = error.code if isinstance(error, CleanerError) else error_to_code(error)
    payload = {
        "type": "error",
        "job_id": job_id,
        "schema_version": SCHEMA_VERSION,
        "code": code,
        "message": str(error),
    }
    if jsonl:
        typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        typer.echo(json.dumps({"error": payload}, ensure_ascii=False), err=True)


def error_to_code(error: Exception) -> str:
    if isinstance(error, CleanerError):
        return error.code
    message = str(error).lower()
    if "input" in message and "not found" in message:
        return "INPUT_NOT_FOUND"
    if "mask" in message and "parse" in message:
        return "MASK_REFINE_FAILED"
    if "lama" in message or "provider" in message:
        return "CLEANER_PROVIDER_NOT_AVAILABLE"
    return "CLEANER_EXECUTION_FAILED"
