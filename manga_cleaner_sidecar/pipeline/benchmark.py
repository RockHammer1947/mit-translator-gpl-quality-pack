from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from manga_cleaner_sidecar.contracts import (
    CleanImageRequest,
    CleanerBenchmarkGate,
    CleanerBenchmarkRequest,
    CleanerConfig,
    CleanerError,
)
from manga_cleaner_sidecar.pipeline.clean_image import clean_image
from manga_cleaner_sidecar.pipeline.model_manager import inspect_lama_large_model, prepare_lama_large_model


PRESET_INPAINTING_SIZE = {
    "fast": 1024,
    "balanced": 1536,
    "quality": 2048,
}


def run_cleaner_benchmark(
    request: CleanerBenchmarkRequest,
    *,
    output_dir: Path,
    prepare_models: bool = False,
) -> dict[str, Any]:
    if not request.items:
        raise CleanerError("INVALID_BENCHMARK_REQUEST", "Cleaner benchmark requires at least one item")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_info = inspect_lama_large_model(request.config.model_path if request.config else None)
    if prepare_models and request.provider == "lama-large-internal":
        model_info = prepare_lama_large_model(request.config.model_path if request.config else None)

    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []
    failures = 0
    for item in request.items:
        for preset in request.presets:
            item_result = _run_one_item(request, item_id=item.id, preset=preset, output_dir=output_dir)
            if item_result["status"] != "completed":
                failures += 1
            results.append(item_result)

    summary = _build_summary(results, request.gate)
    status = "completed" if failures == 0 else "partial_failure"
    if summary["gate_passed"] is False:
        status = "failed" if failures == len(results) else "partial_failure"
    return {
        "schema_version": "manga_cleaner_benchmark_report.v1",
        "job_id": request.job_id,
        "status": status,
        "provider": request.provider,
        "presets": request.presets,
        "model": model_info,
        "gate": request.gate.model_dump(),
        "summary": {
            **summary,
            "duration_ms": int((time.perf_counter() - started_at) * 1000),
        },
        "items": results,
    }


def _run_one_item(
    request: CleanerBenchmarkRequest,
    *,
    item_id: str,
    preset: str,
    output_dir: Path,
) -> dict[str, Any]:
    item = next(entry for entry in request.items if entry.id == item_id)
    item_dir = output_dir / item.id / preset
    item_dir.mkdir(parents=True, exist_ok=True)
    config = _config_for_preset(request, preset)
    clean_request = CleanImageRequest(
        job_id=f"{request.job_id}_{item.id}_{preset}",
        source_image=item.source_image,
        raw_mask_image=item.raw_mask_image,
        detector_refined_mask_image=item.detector_refined_mask_image,
        mask_output=item_dir / "text_mask_refined.png",
        refined_mask_output=item_dir / "text_mask_refined.png",
        mask_debug_output=item_dir / "mask_debug_overlay.png",
        inpaint_quality_report_output=item_dir / "inpaint_quality_report.json",
        cleaned_output=item_dir / "cleaned_image.png",
        manifest_output=item_dir / "cleaner_manifest.json",
        config=config,
        blocks=_load_blocks(item.blocks_path),
    )
    try:
        stats = clean_image(clean_request)
        metrics = stats.get("inpaint_quality_report", {}).get("metrics", {})
        return {
            "id": item.id,
            "preset": preset,
            "status": "completed",
            "provider": config.provider,
            "artifacts": {
                "cleaned_image": str(clean_request.cleaned_output),
                "mask": str(clean_request.mask_output),
                "mask_debug_overlay": str(clean_request.mask_debug_output),
                "inpaint_quality_report": str(clean_request.inpaint_quality_report_output),
                "manifest": str(clean_request.manifest_output),
            },
            "metrics": metrics,
            "gate": _evaluate_gate(metrics, request.gate),
            "metadata": item.metadata,
        }
    except Exception as error:
        code = error.code if isinstance(error, CleanerError) else "CLEANER_EXECUTION_FAILED"
        return {
            "id": item.id,
            "preset": preset,
            "status": "failed",
            "provider": config.provider,
            "error_code": code,
            "error_message": str(error),
            "metadata": item.metadata,
        }


def _config_for_preset(request: CleanerBenchmarkRequest, preset: str) -> CleanerConfig:
    base = request.config.model_copy(deep=True) if request.config else CleanerConfig(provider=request.provider)
    mask_refine_mode = base.mask_refine_mode
    mask_source = base.mask_source
    if request.provider.startswith("lama-large") or preset == "quality":
        if mask_refine_mode == "polygon_cc_refine":
            mask_refine_mode = "mit_fit_text"
        if mask_source == "auto":
            mask_source = "union"
    return base.model_copy(
        update={
            "provider": request.provider,
            "quality_preset": preset,
            "inpainting_size": PRESET_INPAINTING_SIZE[preset],
            "mask_refine_mode": mask_refine_mode,
            "mask_source": mask_source,
        }
    )


def _load_blocks(path: Path | None):
    if path is None:
        return []
    if not path.exists():
        raise CleanerError("INPUT_NOT_FOUND", f"Blocks JSON not found: {path}")
    import json

    from manga_cleaner_sidecar.contracts import CleanerBlock

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_blocks = payload.get("blocks") if isinstance(payload, dict) else payload
    if raw_blocks is None:
        return []
    if not isinstance(raw_blocks, list):
        raise CleanerError("INVALID_BLOCKS", "Blocks JSON must be a list or contain a blocks array")
    return [CleanerBlock.model_validate(block) for block in raw_blocks]


def _evaluate_gate(metrics: dict[str, Any], gate: CleanerBenchmarkGate) -> dict[str, Any]:
    text_residual = _number(metrics.get("text_residual_ratio"))
    background_damage = _number(metrics.get("background_damage_ratio"))
    edge_halo = _number(metrics.get("edge_halo_score"))
    quality_score = _number(metrics.get("quality_score"))
    checks = {
        "text_residual_ratio": text_residual is not None and text_residual <= gate.text_residual_ratio,
        "text_residual_warn_ratio": text_residual is not None and text_residual <= gate.text_residual_warn_ratio,
        "background_damage_ratio": background_damage is not None and background_damage <= gate.background_damage_ratio,
        "edge_halo_score": edge_halo is not None and edge_halo <= gate.edge_halo_score,
        "quality_score": quality_score is not None and quality_score >= gate.quality_score,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def _build_summary(results: list[dict[str, Any]], gate: CleanerBenchmarkGate) -> dict[str, Any]:
    completed = [item for item in results if item.get("status") == "completed"]
    failed = len(results) - len(completed)
    gates = [item.get("gate", {}).get("passed") for item in completed]
    quality_scores = [_number(item.get("metrics", {}).get("quality_score")) for item in completed]
    residuals = [_number(item.get("metrics", {}).get("text_residual_ratio")) for item in completed]
    damages = [_number(item.get("metrics", {}).get("background_damage_ratio")) for item in completed]
    return {
        "run_count": len(results),
        "completed_count": len(completed),
        "failed_count": failed,
        "gate_passed": bool(completed) and failed == 0 and all(gates),
        "gate_target": gate.model_dump(),
        "avg_quality_score": _average(quality_scores),
        "max_text_residual_ratio": _maximum(residuals),
        "max_background_damage_ratio": _maximum(damages),
    }


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _average(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _maximum(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None
