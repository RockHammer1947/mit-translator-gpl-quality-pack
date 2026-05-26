from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ocr_sidecar.contracts import BatchRequest
from ocr_sidecar.engines import get_engine
from ocr_sidecar.protocol import avg_confidence, dump_json, to_artifact_blocks


DEFAULT_SWEEP = {
    "crop_policies": ["mit_textline", "adaptive", "bbox", "polygon_perspective"],
    "target_text_heights": [32, 48, 64, 96],
    "padding_ratios": [0.1, 0.18],
    "preprocess": ["none", "auto"],
}


def run_manga_ocr_benchmark(
    *,
    payload: dict[str, Any],
    output_dir: Path,
    engine: str | None,
    language_hint: str | None,
    job_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    request = BatchRequest.model_validate(payload)
    resolved_engine = engine or request.engine
    resolved_language = language_hint or request.effective_language_hint()
    ocr_engine = get_engine(resolved_engine)
    sweep = _normalize_sweep(payload.get("sweep"))
    started_at = time.perf_counter()
    runs: list[dict[str, Any]] = []
    for combo_index, combo in enumerate(_sweep_combinations(sweep), 1):
        combo_id = f"run_{combo_index:03d}"
        debug_dir = output_dir / "debug_crops" / combo_id
        result_items: list[dict[str, Any]] = []
        recognized_items = 0
        block_count = 0
        confidences: list[float] = []
        for item in request.items:
            try:
                raw_blocks = ocr_engine.recognize_image(
                    image_path=item.image_path,
                    lang_hint=item.lang_hint or resolved_language,
                    region=item.region,
                    polygon=item.polygon,
                    crop_policy=str(combo["crop_policy"]),
                    target_text_height=int(combo["target_text_height"]),
                    padding_ratio=float(combo["padding_ratio"]),
                    preprocess=str(combo["preprocess"]),
                    debug_output_dir=debug_dir,
                    metadata={**item.metadata, "ocr_batch_item_id": item.id, "benchmark_combo_id": combo_id},
                )
                blocks = to_artifact_blocks(raw_blocks, engine=resolved_engine, language_hint=item.lang_hint or resolved_language)
                if blocks:
                    recognized_items += 1
                block_count += len(blocks)
                confidences.append(avg_confidence(blocks))
                result_items.append({"id": item.id, "status": "completed", "blocks": [block.model_dump(mode="json") for block in blocks]})
            except Exception as error:
                result_items.append({"id": item.id, "status": "failed", "error_message": str(error), "blocks": []})
        item_count = len(request.items)
        runs.append(
            {
                "id": combo_id,
                "combo": combo,
                "summary": {
                    "item_count": item_count,
                    "recognized_item_count": recognized_items,
                    "recognized_item_ratio": recognized_items / item_count if item_count else 0.0,
                    "block_count": block_count,
                    "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                },
                "debug_crops": str(debug_dir) if debug_dir.exists() else None,
                "items": result_items,
            }
        )
    best = sorted(
        runs,
        key=lambda run: (
            run["summary"]["recognized_item_ratio"],
            run["summary"]["block_count"],
            run["summary"]["avg_confidence"],
        ),
        reverse=True,
    )[0] if runs else None
    report = {
        "schema_version": "ocr_manga_benchmark_report.v1",
        "job_id": job_id,
        "engine": resolved_engine,
        "language_hint": resolved_language,
        "source": payload.get("source"),
        "summary": {
            "run_count": len(runs),
            "item_count": len(request.items),
            "best_run_id": best["id"] if best else None,
            "best_recognized_item_ratio": best["summary"]["recognized_item_ratio"] if best else 0.0,
            "best_block_count": best["summary"]["block_count"] if best else 0,
            "duration_ms": int((time.perf_counter() - started_at) * 1000),
        },
        "best_run": best,
        "runs": runs,
    }
    dump_json(output_dir / "ocr_benchmark_report.json", report)
    return report


def _normalize_sweep(raw: Any) -> dict[str, list[Any]]:
    if not isinstance(raw, dict):
        return DEFAULT_SWEEP
    return {
        "crop_policies": _list_or_default(raw.get("crop_policies"), DEFAULT_SWEEP["crop_policies"]),
        "target_text_heights": _list_or_default(raw.get("target_text_heights"), DEFAULT_SWEEP["target_text_heights"]),
        "padding_ratios": _list_or_default(raw.get("padding_ratios"), DEFAULT_SWEEP["padding_ratios"]),
        "preprocess": _list_or_default(raw.get("preprocess"), DEFAULT_SWEEP["preprocess"]),
    }


def _list_or_default(value: Any, fallback: list[Any]) -> list[Any]:
    if isinstance(value, list) and value:
        return value
    return fallback


def _sweep_combinations(sweep: dict[str, list[Any]]):
    for crop_policy in sweep["crop_policies"]:
        for target_text_height in sweep["target_text_heights"]:
            for padding_ratio in sweep["padding_ratios"]:
                for preprocess in sweep["preprocess"]:
                    yield {
                        "crop_policy": crop_policy,
                        "target_text_height": target_text_height,
                        "padding_ratio": padding_ratio,
                        "preprocess": preprocess,
                    }
