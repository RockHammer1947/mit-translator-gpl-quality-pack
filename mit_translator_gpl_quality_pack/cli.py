from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from mit_translator_gpl_quality_pack import __version__

SCHEMA_VERSION = "mit-translator-gpl-quality-pack.v1"
LICENSE = "GPL-3.0-only"
LICENSE_CLASS = "copyleft_gpl"

app = typer.Typer(
    name="mit-translator-gpl-quality-pack",
    help="Optional GPL-3.0 quality provider pack for manga translation",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("provider-manifest")
def provider_manifest(jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL")) -> None:
    _emit(_manifest(), jsonl)


@app.command("doctor")
def doctor(jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL")) -> None:
    payload = {
        "type": "doctor",
        "job_id": "doctor",
        "schema_version": SCHEMA_VERSION,
        "pack_id": "mit-translator-gpl-quality-pack",
        "pack_version": __version__,
        "license": LICENSE,
        "license_class": LICENSE_CLASS,
        "manual_enable_required": True,
        "providers": [
            _doctor_delegate("mit-ctd", "comic_detector", ["doctor", "--jsonl"], provider_name="mit-ctd"),
            _doctor_delegate("mit-48px-ocr", "ocr", ["doctor", "--jsonl"], engine_name="mit-48px-internal"),
            _doctor_delegate("mit-layout-reference", "layout", ["doctor", "--jsonl"], provider_name="native_layout"),
        ],
        "manifest": _manifest(),
    }
    payload["available"] = all(provider.get("available") for provider in payload["providers"])
    _emit(payload, jsonl)


@app.command("detect-image")
def detect_image(
    input_path: Path = typer.Option(..., "--input", help="Input image path"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output directory"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Manifest output path"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL"),
    max_regions: int = typer.Option(24, "--max-regions", help="Maximum regions"),
    coverage_preset: str = typer.Option("balanced", "--coverage-preset", help="Coverage preset"),
) -> None:
    resolved_job_id = job_id or "gpl_detect"
    args = [
        "detect-image",
        "--input",
        str(input_path),
        "--provider",
        "mit-ctd",
        "--output-dir",
        str(output_dir),
        "--job-id",
        resolved_job_id,
        "--max-regions",
        str(max_regions),
        "--coverage-preset",
        coverage_preset,
    ]
    if manifest:
        args += ["--manifest", str(manifest)]
    if jsonl:
        args.append("--jsonl")
    _run_stage("comic_detector", args, jsonl=jsonl, job_id=resolved_job_id, stage="detect-image")


@app.command("recognize-batch")
def recognize_batch(
    input_path: Path = typer.Option(..., "--input", "--input-json", help="OCR batch request JSON"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output directory"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Manifest output path"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Job id"),
    language_hint: Optional[str] = typer.Option(None, "--language-hint", "--lang-hint", help="Language hint"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL"),
) -> None:
    resolved_job_id = job_id or "gpl_ocr"
    engine = os.environ.get("GPL_QUALITY_PACK_OCR_ENGINE", "mit-48px-internal")
    args = [
        "recognize-batch",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--engine",
        engine,
        "--job-id",
        resolved_job_id,
    ]
    if language_hint:
        args += ["--language-hint", language_hint]
    if manifest:
        args += ["--manifest", str(manifest)]
    if jsonl:
        args.append("--jsonl")
    _run_stage("ocr", args, jsonl=jsonl, job_id=resolved_job_id, stage="recognize-batch")


@app.command("merge-textlines")
def merge_textlines(
    input_path: Path = typer.Option(..., "--input", help="manga_layout_request.v1 JSON file"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output directory"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Manifest output path"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL"),
) -> None:
    resolved_job_id = job_id or "gpl_layout"
    args = [
        "merge-textlines",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--job-id",
        resolved_job_id,
    ]
    if manifest:
        args += ["--manifest", str(manifest)]
    if jsonl:
        args.append("--jsonl")
    _run_stage("layout", args, jsonl=jsonl, job_id=resolved_job_id, stage="merge-textlines")


def _manifest() -> dict:
    return {
        "type": "provider_manifest",
        "job_id": "provider_manifest",
        "schema_version": SCHEMA_VERSION,
        "pack_id": "mit-translator-gpl-quality-pack",
        "pack_version": __version__,
        "license": LICENSE,
        "license_class": LICENSE_CLASS,
        "manual_enable_required": True,
        "capabilities": [
            {
                "id": "mit-ctd",
                "kind": "detector",
                "command": "detect-image",
                "schema_version": "comic-detector-sidecar.v1",
                "required_models": ["comic-detector.mit-ctd.onnx"],
            },
            {
                "id": "mit-48px-ocr",
                "kind": "ocr",
                "command": "recognize-batch",
                "schema_version": "ocr-sidecar.v1",
                "required_models": ["ocr.mit-48px"],
            },
            {
                "id": "mit-layout-reference",
                "kind": "layout",
                "command": "merge-textlines",
                "schema_version": "manga-layout-sidecar.v1",
                "required_models": [],
            },
        ],
    }


def _doctor_delegate(
    provider_id: str,
    tool: str,
    args: list[str],
    *,
    provider_name: str | None = None,
    engine_name: str | None = None,
) -> dict:
    base = {
        "id": provider_id,
        "available": False,
        "license": LICENSE,
        "license_class": LICENSE_CLASS,
        "manual_enable_required": True,
    }
    try:
        completed = _run_delegate(tool, args, timeout=30)
    except Exception as error:
        return {**base, "reason": str(error)}
    if completed.returncode != 0:
        return {**base, "reason": completed.stderr.strip() or completed.stdout.strip() or "delegate doctor failed"}
    payload = _parse_json_payload(completed.stdout)
    if provider_name:
        providers = payload.get("providers") if isinstance(payload, dict) else None
        provider = next((item for item in providers or [] if item.get("name") == provider_name or item.get("provider") == provider_name), None)
        if provider:
            return {**base, "available": bool(provider.get("available")), "delegate": provider}
    if engine_name:
        engines = payload.get("engines") if isinstance(payload, dict) else None
        engine = next((item for item in engines or [] if item.get("name") == engine_name), None)
        if engine:
            return {**base, "available": bool(engine.get("available")) and engine.get("model_ready") is not False, "delegate": engine}
    if isinstance(payload, dict):
        return {**base, "available": bool(payload.get("available", True)), "delegate": payload}
    return {**base, "reason": "delegate doctor returned no JSON payload"}


def _run_stage(tool: str, args: list[str], *, jsonl: bool, job_id: str, stage: str) -> None:
    _emit(
        {
            "type": "started",
            "job_id": job_id,
            "schema_version": SCHEMA_VERSION,
            "pack_version": __version__,
            "stage": stage,
            "license": LICENSE,
            "license_class": LICENSE_CLASS,
        },
        jsonl,
    )
    completed = _run_delegate(tool, args, timeout=None)
    if completed.stdout:
        sys.stdout.write(completed.stdout)
        if not completed.stdout.endswith("\n"):
            sys.stdout.write("\n")
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        _emit(
            {
                "type": "error",
                "job_id": job_id,
                "schema_version": SCHEMA_VERSION,
                "code": "GPL_QUALITY_PACK_DELEGATE_FAILED",
                "message": completed.stderr.strip() or completed.stdout.strip() or f"{stage} failed",
            },
            jsonl,
        )
        raise typer.Exit(completed.returncode)
    _emit(
        {
            "type": "done",
            "job_id": job_id,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "stage": stage,
        },
        jsonl,
    )


def _run_delegate(tool: str, args: list[str], *, timeout: int | None) -> subprocess.CompletedProcess[str]:
    command, cwd = _delegate_command(tool)
    return subprocess.run(
        [*command, *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _delegate_command(tool: str) -> tuple[list[str], Path | None]:
    env_key = {
        "comic_detector": "GPL_QUALITY_PACK_COMIC_DETECTOR_CMD",
        "ocr": "GPL_QUALITY_PACK_OCR_CMD",
        "layout": "GPL_QUALITY_PACK_LAYOUT_CMD",
    }[tool]
    override = os.environ.get(env_key)
    if override:
        return shlex.split(override), None

    module = {
        "comic_detector": "comic_detector_sidecar.cli",
        "ocr": "ocr_sidecar.cli",
        "layout": "manga_layout_sidecar.cli",
    }[tool]
    return [sys.executable, "-m", module], None

    binary = {
        "comic_detector": "comic-detector-sidecar",
        "ocr": "ocr-sidecar",
        "layout": "manga-layout-sidecar",
    }[tool]
    resolved = shutil.which(binary)
    if not resolved:
        raise FileNotFoundError(f"{binary} not found. Set {env_key} to a compatible delegate command.")
    return [resolved], None


def _parse_json_payload(stdout: str) -> dict:
    for line in stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _emit(payload: dict, jsonl: bool) -> None:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":") if jsonl else None, indent=None if jsonl else 2)
    print(text, flush=True)


if __name__ == "__main__":
    app()
