from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from manga_cleaner_sidecar.contracts import (
    CleanBatchRequest,
    CleanImageRequest,
    CleanerBlock,
    CleanerConfig,
    CleanerError,
    CleanerManifest,
)
from manga_cleaner_sidecar.jsonl import dump_json, emit, emit_started, make_job_id, print_error
from manga_cleaner_sidecar.pipeline.benchmark import run_cleaner_benchmark
from manga_cleaner_sidecar.pipeline.clean_image import build_mask_only, clean_image
from manga_cleaner_sidecar.pipeline.doctor import run_doctor
from manga_cleaner_sidecar.pipeline.model_manager import inspect_lama_large_model, prepare_lama_large_model

app = typer.Typer(
    name="manga-cleaner-sidecar",
    help="Manga text mask refinement and image cleaning sidecar",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("doctor")
def doctor(
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL to stdout"),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="Optional LaMa model path to check"),
    lama_command: Optional[str] = typer.Option(None, "--lama-command", help="Optional external LaMa adapter command"),
) -> None:
    payload = run_doctor(model_path=model_path, lama_command=lama_command)
    typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


@app.command("prepare-models")
def prepare_models_cmd(
    provider: str = typer.Option("lama-large-internal", "--provider", help="Model provider to prepare"),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="Optional target model path"),
    force: bool = typer.Option(False, "--force", help="Re-download even if a model already exists"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events"),
) -> None:
    resolved_job_id = job_id or make_job_id("job_manga_cleaner_prepare")
    try:
        if provider != "lama-large-internal":
            raise CleanerError("CLEANER_PROVIDER_NOT_FOUND", f"prepare-models only supports lama-large-internal, got {provider}")
        emit_started(resolved_job_id, provider, jsonl)
        emit({"type": "progress", "job_id": resolved_job_id, "stage": "prepare_models", "progress": 0.1, "message": "checking LaMa model"}, jsonl)
        before = inspect_lama_large_model(model_path)
        if before["exists"] and before["hash_ok"] and not force:
            result = {**before, "downloaded": False, "status": "ready"}
        else:
            emit({"type": "progress", "job_id": resolved_job_id, "stage": "prepare_models", "progress": 0.35, "message": "downloading LaMa model"}, jsonl)
            result = prepare_lama_large_model(model_path, force=force)
        emit({"type": "done", "job_id": resolved_job_id, "status": "completed", "result": result}, jsonl)
        if not jsonl:
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        print_error(error, resolved_job_id, jsonl)
        raise typer.Exit(1) from error


@app.command("build-mask")
def build_mask_cmd(
    input_path: Path = typer.Option(..., "--input", help="Source manga image"),
    blocks_path: Optional[Path] = typer.Option(None, "--blocks", help="Translated/text blocks JSON"),
    raw_mask_image: Optional[Path] = typer.Option(None, "--raw-mask", help="Detector raw text mask"),
    detector_refined_mask_image: Optional[Path] = typer.Option(None, "--detector-refined-mask", help="Detector refined text mask"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for artifacts"),
    mask_output: Optional[Path] = typer.Option(None, "--mask-output", help="Path for text_mask_refined.png"),
    mask_debug_output: Optional[Path] = typer.Option(None, "--mask-debug-output", help="Path for mask debug overlay"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Path for cleaner manifest"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events"),
    mask_refine_mode: str = typer.Option("mit_fit_text", "--mask-refine-mode", help="mit_fit_text|polygon_cc_refine|raw_mask|threshold|disabled"),
    mask_source: str = typer.Option("union", "--mask-source", help="auto|raw|refined|union|threshold"),
    mask_dilation_offset: int = typer.Option(18, "--mask-dilation-offset"),
    kernel_size: int = typer.Option(3, "--kernel-size"),
) -> None:
    resolved_job_id = job_id or make_job_id("job_manga_mask")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        config = CleanerConfig(
            provider="none",
            mask_refine_mode=mask_refine_mode,  # type: ignore[arg-type]
            mask_source=mask_source,  # type: ignore[arg-type]
            mask_dilation_offset=mask_dilation_offset,
            kernel_size=kernel_size,
        )
        request = _request_from_cli(
            job_id=resolved_job_id,
            input_path=input_path,
            blocks_path=blocks_path,
            raw_mask_image=raw_mask_image,
            detector_refined_mask_image=detector_refined_mask_image,
            output_dir=output_dir,
            provider="none",
            config=config,
            mask_output=mask_output,
            cleaned_output=output_dir / "cleaned_image.png",
            manifest=manifest,
            mask_debug_output=mask_debug_output,
        )
        emit_started(resolved_job_id, "mask-only", jsonl)
        emit({"type": "progress", "job_id": resolved_job_id, "stage": "mask", "progress": 0.4, "message": "Refining manga text mask"}, jsonl)
        stats = build_mask_only(request)
        _write_manifest_and_events(request, stats, jsonl, cleaned=False)
    except Exception as error:
        print_error(error, resolved_job_id, jsonl)
        raise typer.Exit(1) from error


@app.command("clean-image")
def clean_image_cmd(
    input_path: Path = typer.Option(..., "--input", help="Source manga image"),
    blocks_path: Optional[Path] = typer.Option(None, "--blocks", help="Translated/text blocks JSON"),
    raw_mask_image: Optional[Path] = typer.Option(None, "--raw-mask", help="Detector raw text mask"),
    detector_refined_mask_image: Optional[Path] = typer.Option(None, "--detector-refined-mask", help="Detector refined text mask"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for artifacts"),
    provider: str = typer.Option("telea", "--provider", help="lama-large-internal|telea|lama-large|none"),
    quality_preset: str = typer.Option("balanced", "--quality-preset", help="fast|balanced|quality"),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="Optional LaMa model path"),
    lama_command: Optional[str] = typer.Option(None, "--lama-command", help="External LaMa adapter command"),
    mask_output: Optional[Path] = typer.Option(None, "--mask-output", help="Path for text_mask_refined.png"),
    cleaned_output: Optional[Path] = typer.Option(None, "--cleaned-output", help="Path for cleaned image"),
    mask_debug_output: Optional[Path] = typer.Option(None, "--mask-debug-output", help="Path for mask debug overlay"),
    inpaint_quality_report_output: Optional[Path] = typer.Option(None, "--inpaint-quality-report", help="Path for inpaint_quality_report.json"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Path for cleaner manifest"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events"),
    mask_refine_mode: str = typer.Option("auto", "--mask-refine-mode", help="auto|mit_fit_text|polygon_cc_refine|raw_mask|threshold|disabled"),
    mask_source: str = typer.Option("auto", "--mask-source", help="auto|raw|refined|union|threshold"),
    mask_dilation_offset: int = typer.Option(18, "--mask-dilation-offset"),
    kernel_size: int = typer.Option(3, "--kernel-size"),
    inpaint_radius: float = typer.Option(3.0, "--inpaint-radius"),
    inpainting_size: int = typer.Option(2048, "--inpainting-size"),
    device: Optional[str] = typer.Option(None, "--device"),
) -> None:
    resolved_job_id = job_id or make_job_id("job_manga_clean")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        config = CleanerConfig(
            provider=provider,  # type: ignore[arg-type]
            quality_preset=quality_preset,  # type: ignore[arg-type]
            mask_refine_mode=_resolve_cli_mask_refine_mode(provider, quality_preset, mask_refine_mode),  # type: ignore[arg-type]
            mask_source=_resolve_cli_mask_source(provider, quality_preset, mask_source),  # type: ignore[arg-type]
            mask_dilation_offset=mask_dilation_offset,
            kernel_size=kernel_size,
            inpaint_radius=inpaint_radius,
            inpainting_size=inpainting_size,
            model_path=model_path,
            lama_command=lama_command,
            device=device,
        )
        request = _request_from_cli(
            job_id=resolved_job_id,
            input_path=input_path,
            blocks_path=blocks_path,
            raw_mask_image=raw_mask_image,
            detector_refined_mask_image=detector_refined_mask_image,
            output_dir=output_dir,
            provider=provider,
            config=config,
            mask_output=mask_output,
            cleaned_output=cleaned_output,
            manifest=manifest,
            mask_debug_output=mask_debug_output,
            inpaint_quality_report_output=inpaint_quality_report_output,
        )
        emit_started(resolved_job_id, provider, jsonl)
        emit({"type": "progress", "job_id": resolved_job_id, "stage": "mask", "progress": 0.35, "message": "Refining manga text mask"}, jsonl)
        stats = clean_image(request)
        emit({"type": "progress", "job_id": resolved_job_id, "stage": "clean", "progress": 0.9, "message": "Writing cleaner artifacts"}, jsonl)
        _write_manifest_and_events(request, stats, jsonl, cleaned=True)
    except Exception as error:
        print_error(error, resolved_job_id, jsonl)
        raise typer.Exit(1) from error


@app.command("clean-batch")
def clean_batch_cmd(
    input_path: Path = typer.Option(..., "--input", help="Batch request JSON"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for batch artifacts"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Override batch job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events"),
) -> None:
    resolved_job_id = job_id or make_job_id("job_manga_clean_batch")
    try:
        if not input_path.exists():
            raise CleanerError("INPUT_NOT_FOUND", f"Batch request not found: {input_path}")
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        batch = CleanBatchRequest.model_validate({**payload, "job_id": resolved_job_id if job_id else payload.get("job_id", resolved_job_id)})
        output_dir.mkdir(parents=True, exist_ok=True)
        emit_started(batch.job_id, batch.config.provider, jsonl)
        items: list[dict[str, object]] = []
        failures = 0
        for index, item in enumerate(batch.items, 1):
            item_dir = output_dir / item.id
            item_dir.mkdir(parents=True, exist_ok=True)
            emit(
                {
                    "type": "progress",
                    "job_id": batch.job_id,
                    "stage": "clean_batch",
                    "progress": index / max(1, len(batch.items)),
                    "message": f"Cleaning image {index}/{len(batch.items)}",
                    "item_id": item.id,
                },
                jsonl,
            )
            try:
                item_config = _normalize_config_for_quality(item.config or batch.config)
                request = _request_from_cli(
                    job_id=f"{batch.job_id}_{item.id}",
                    input_path=item.source_image,
                    blocks_path=item.blocks_path,
                    raw_mask_image=item.raw_mask_image,
                    detector_refined_mask_image=item.detector_refined_mask_image,
                    output_dir=item_dir,
                    provider=item_config.provider,
                    config=item_config,
                    mask_output=item_dir / "text_mask_refined.png",
                    cleaned_output=item_dir / "cleaned_image.png",
                    manifest=item_dir / "cleaner_manifest.json",
                    mask_debug_output=item_dir / "mask_debug_overlay.png",
                    inpaint_quality_report_output=item_dir / "inpaint_quality_report.json",
                )
                stats = clean_image(request)
                _write_manifest_and_events(request, stats, jsonl, cleaned=True)
                items.append({"id": item.id, "status": "completed", "manifest": str(request.manifest_output), "metadata": item.metadata})
            except Exception as error:
                failures += 1
                code = error.code if isinstance(error, CleanerError) else "CLEANER_EXECUTION_FAILED"
                emit({"type": "error", "job_id": batch.job_id, "item_id": item.id, "code": code, "message": str(error)}, jsonl)
                items.append({"id": item.id, "status": "failed", "error_code": code, "error_message": str(error), "metadata": item.metadata})
        manifest_path = output_dir / "cleaner_batch_manifest.json"
        manifest_doc = {
            "schema_version": "manga-cleaner-sidecar.v1",
            "job_id": batch.job_id,
            "status": "partial_failure" if failures else "completed",
            "provider": batch.config.provider,
            "artifacts": {"manifest": str(manifest_path)},
            "summary": {"image_count": len(batch.items), "failed_count": failures},
            "items": items,
        }
        dump_json(manifest_path, manifest_doc, manifest=True)
        emit({"type": "artifact", "job_id": batch.job_id, "kind": "cleaner_batch_manifest", "path": str(manifest_path)}, jsonl)
        emit({"type": "done", "job_id": batch.job_id, "status": manifest_doc["status"], "manifest_path": str(manifest_path), "result": manifest_doc}, jsonl)
        if not jsonl:
            typer.echo(json.dumps(manifest_doc, ensure_ascii=False, indent=2))
    except Exception as error:
        print_error(error, resolved_job_id, jsonl)
        raise typer.Exit(1) from error


@app.command("benchmark")
def benchmark_cmd(
    input_path: Path = typer.Option(..., "--input", help="Cleaner benchmark request JSON"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for benchmark artifacts"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Path for benchmark manifest"),
    prepare_models: bool = typer.Option(False, "--prepare-models/--no-prepare-models", help="Download/check model before running"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Override benchmark job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events"),
) -> None:
    resolved_job_id = job_id or make_job_id("job_manga_cleaner_benchmark")
    try:
        if not input_path.exists():
            raise CleanerError("INPUT_NOT_FOUND", f"Benchmark request not found: {input_path}")
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        from manga_cleaner_sidecar.contracts import CleanerBenchmarkRequest

        request = CleanerBenchmarkRequest.model_validate(
            {**payload, "job_id": resolved_job_id if job_id else payload.get("job_id", resolved_job_id)}
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "cleaner_benchmark_report.json"
        manifest_path = manifest or (output_dir / "cleaner_benchmark_manifest.json")
        emit_started(request.job_id, request.provider, jsonl)
        emit(
            {
                "type": "progress",
                "job_id": request.job_id,
                "stage": "benchmark",
                "progress": 0.1,
                "message": "Running cleaner benchmark",
            },
            jsonl,
        )
        report = run_cleaner_benchmark(request, output_dir=output_dir, prepare_models=prepare_models)
        dump_json(report_path, report)
        manifest_doc = {
            "schema_version": "manga-cleaner-sidecar.v1",
            "job_id": request.job_id,
            "status": report["status"],
            "provider": request.provider,
            "artifacts": {
                "benchmark_report": str(report_path),
                "manifest": str(manifest_path),
            },
            "summary": report["summary"],
            "warnings": _benchmark_warnings(report),
        }
        dump_json(manifest_path, manifest_doc, manifest=True)
        emit({"type": "artifact", "job_id": request.job_id, "kind": "cleaner_benchmark_report", "path": str(report_path)}, jsonl)
        emit({"type": "artifact", "job_id": request.job_id, "kind": "cleaner_benchmark_manifest", "path": str(manifest_path)}, jsonl)
        emit({"type": "done", "job_id": request.job_id, "status": report["status"], "manifest_path": str(manifest_path), "result": manifest_doc}, jsonl)
        if not jsonl:
            typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as error:
        print_error(error, resolved_job_id, jsonl)
        raise typer.Exit(1) from error


def _request_from_cli(
    *,
    job_id: str,
    input_path: Path,
    blocks_path: Path | None,
    raw_mask_image: Path | None,
    detector_refined_mask_image: Path | None,
    output_dir: Path,
    provider: str,
    config: CleanerConfig,
    mask_output: Path | None,
    cleaned_output: Path | None,
    manifest: Path | None,
    mask_debug_output: Path | None,
    inpaint_quality_report_output: Path | None = None,
) -> CleanImageRequest:
    if not input_path.exists():
        raise CleanerError("INPUT_NOT_FOUND", f"Input image not found: {input_path}")
    blocks = _load_blocks(blocks_path)
    return CleanImageRequest(
        job_id=job_id,
        source_image=input_path,
        raw_mask_image=raw_mask_image,
        detector_refined_mask_image=detector_refined_mask_image,
        mask_output=mask_output or (output_dir / "text_mask_refined.png"),
        refined_mask_output=output_dir / "text_mask_refined.png",
        mask_debug_output=mask_debug_output or (output_dir / "mask_debug_overlay.png"),
        inpaint_quality_report_output=inpaint_quality_report_output or (output_dir / "inpaint_quality_report.json"),
        cleaned_output=cleaned_output or (output_dir / "cleaned_image.png"),
        manifest_output=manifest or (output_dir / "cleaner_manifest.json"),
        config=config,
        blocks=blocks,
    )


def _load_blocks(path: Path | None) -> list[CleanerBlock]:
    if path is None:
        return []
    if not path.exists():
        raise CleanerError("INPUT_NOT_FOUND", f"Blocks JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_blocks = payload.get("blocks") if isinstance(payload, dict) else payload
    if raw_blocks is None:
        return []
    if not isinstance(raw_blocks, list):
        raise CleanerError("INVALID_BLOCKS", "Blocks JSON must be a list or contain a blocks array")
    return [CleanerBlock.model_validate(block) for block in raw_blocks]


def _write_manifest_and_events(request: CleanImageRequest, stats: dict, jsonl: bool, *, cleaned: bool) -> None:
    artifacts = {
        "mask": str(request.mask_output),
        "refined_mask": str(request.refined_mask_output or request.mask_output),
        "manifest": str(request.manifest_output),
    }
    if request.mask_debug_output is not None:
        artifacts["mask_debug_overlay"] = str(request.mask_debug_output)
    if cleaned:
        artifacts["cleaned_image"] = str(request.cleaned_output)
    if cleaned and request.inpaint_quality_report_output is not None:
        artifacts["inpaint_quality_report"] = str(request.inpaint_quality_report_output)
        dump_json(request.inpaint_quality_report_output, stats.get("inpaint_quality_report", {}))
    quality_warnings = stats.get("inpaint_quality_report", {}).get("warnings", []) if cleaned else []
    warnings = [{"code": code, "message": code.replace("_", " ").lower()} for code in stats["mask_refinement"].get("warning_codes", [])]
    warnings.extend(quality_warnings)
    manifest_doc = CleanerManifest(
        job_id=request.job_id,
        status="completed",
        provider=request.config.provider if cleaned else "mask-only",
        source={"type": "image", "path": str(request.source_image)},
        artifacts=artifacts,
        model=_cleaner_model_info(request),
        summary={
            "block_count": len(request.blocks),
            "mask_pixels": stats["mask_pixels"],
            "mask_coverage_ratio": stats["mask_refinement"].get("mask_coverage_ratio", 0.0),
            "raw_candidate_pixels": stats["mask_refinement"].get("raw_candidate_pixels", 0),
            "residual_candidate_ratio": stats["mask_refinement"].get("residual_candidate_ratio", 0.0),
            "mask_refinement": stats["mask_refinement"],
            "quality": stats.get("inpaint_quality_report", {}).get("metrics"),
        },
        warnings=warnings,
    )
    dump_json(request.manifest_output, manifest_doc.model_dump(mode="json"), manifest=True)
    emit({"type": "artifact", "job_id": request.job_id, "kind": "text_mask_refined", "path": str(request.mask_output)}, jsonl)
    if request.mask_debug_output is not None:
        emit({"type": "artifact", "job_id": request.job_id, "kind": "mask_debug_overlay", "path": str(request.mask_debug_output)}, jsonl)
    if cleaned:
        emit({"type": "artifact", "job_id": request.job_id, "kind": "cleaned_image", "path": str(request.cleaned_output)}, jsonl)
        if request.inpaint_quality_report_output is not None:
            emit({"type": "artifact", "job_id": request.job_id, "kind": "inpaint_quality_report", "path": str(request.inpaint_quality_report_output)}, jsonl)
    emit({"type": "artifact", "job_id": request.job_id, "kind": "cleaner_manifest", "path": str(request.manifest_output)}, jsonl)
    emit(
        {
            "type": "done",
            "job_id": request.job_id,
            "status": "completed",
            "manifest_path": str(request.manifest_output),
            "result": manifest_doc.model_dump(mode="json"),
        },
        jsonl,
    )
    if not jsonl:
        typer.echo(manifest_doc.model_dump_json(indent=2))


def _cleaner_model_info(request: CleanImageRequest) -> dict[str, object] | None:
    if request.config.provider != "lama-large-internal":
        return None
    info = inspect_lama_large_model(request.config.model_path)
    return {"path": info["path"], "name": info["name"], "hash_ok": info["hash_ok"]}


def _resolve_cli_mask_refine_mode(provider: str, quality_preset: str, value: str) -> str:
    if value != "auto":
        return value
    if provider.startswith("lama-large") or quality_preset == "quality":
        return "mit_fit_text"
    return "polygon_cc_refine"


def _resolve_cli_mask_source(provider: str, quality_preset: str, value: str) -> str:
    if value != "auto":
        return value
    if provider.startswith("lama-large") or quality_preset == "quality":
        return "union"
    return "raw"


def _normalize_config_for_quality(config: CleanerConfig) -> CleanerConfig:
    updates: dict[str, object] = {}
    if config.provider.startswith("lama-large") or config.quality_preset == "quality":
        if config.mask_refine_mode == "polygon_cc_refine":
            updates["mask_refine_mode"] = "mit_fit_text"
        if config.mask_source == "auto":
            updates["mask_source"] = "union"
    return config.model_copy(update=updates) if updates else config


def _benchmark_warnings(report: dict[str, object]) -> list[dict[str, str]]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return []
    if summary.get("gate_passed") is False:
        return [{"code": "CLEANER_BENCHMARK_GATE_FAILED", "message": "Cleaner benchmark did not meet the configured MIT quality gate."}]
    return []


if __name__ == "__main__":
    app()
