from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def build_inpaint_quality_report(
    *,
    job_id: str,
    provider: str,
    source_image: Path,
    cleaned_image: Path,
    mask_image: Path,
    started_at: float,
    peak_rss_mb: float | None = None,
    mask_refinement_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    cleaned = cv2.imread(str(cleaned_image), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_image), cv2.IMREAD_GRAYSCALE)
    if source is None or cleaned is None or mask is None:
        return {
            "schema_version": "manga_inpaint_quality_report.v1",
            "job_id": job_id,
            "provider": provider,
            "status": "failed",
            "metrics": {},
            "warnings": [{"code": "QUALITY_INPUT_UNREADABLE", "message": "quality inputs could not be read"}],
        }
    if cleaned.shape[:2] != source.shape[:2]:
        cleaned = cv2.resize(cleaned, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != source.shape[:2]:
        mask = cv2.resize(mask, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_bool = mask > 0
    image_pixels = max(1, mask.shape[0] * mask.shape[1])
    mask_pixels = int(np.count_nonzero(mask_bool))
    mask_coverage = mask_pixels / image_pixels
    gray_cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    text_residual_ratio = float(np.count_nonzero((gray_cleaned < 90) & mask_bool) / max(1, mask_pixels))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    protected = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2) > 0
    outside = ~protected
    diff = cv2.absdiff(source, cleaned)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    outside_pixels = int(np.count_nonzero(outside))
    background_damage_ratio = float(np.count_nonzero((diff_gray > 14) & outside) / max(1, outside_pixels))

    edge = (cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0) & (~mask_bool)
    edge_halo_score = float(np.mean(diff_gray[edge]) / 255.0) if np.count_nonzero(edge) else 0.0
    mask_stats = mask_refinement_stats or {}
    unmasked_candidate_ratio = _number(mask_stats.get("residual_candidate_ratio")) or 0.0
    final_to_candidate_ratio = _number(mask_stats.get("final_to_candidate_ratio"))
    score = max(
        0.0,
        min(
            1.0,
            1.0
            - text_residual_ratio * 1.8
            - background_damage_ratio * 2.0
            - edge_halo_score * 0.8
            - unmasked_candidate_ratio * 1.3
            - max(0.0, mask_coverage - 0.28) * 1.5,
        ),
    )
    warnings: list[dict[str, str]] = []
    if mask_coverage > 0.35:
        warnings.append({"code": "CATASTROPHIC_OVER_MASK_RISK", "message": "refined mask covers too much of the image"})
    if text_residual_ratio > 0.12:
        warnings.append({"code": "TEXT_RESIDUAL_HIGH", "message": "dark residual pixels remain inside the cleaned mask"})
    if background_damage_ratio > 0.10:
        warnings.append({"code": "BACKGROUND_DAMAGE_HIGH", "message": "pixels outside protected mask changed more than expected"})
    if unmasked_candidate_ratio > 0.15:
        warnings.append({"code": "UNMASKED_TEXT_CANDIDATE_HIGH", "message": "detector candidate text pixels were left outside the cleaner mask"})
    return {
        "schema_version": "manga_inpaint_quality_report.v1",
        "job_id": job_id,
        "provider": provider,
        "status": "completed",
        "metrics": {
            "mask_leak_ratio": max(0.0, mask_coverage - 0.28),
            "text_residual_ratio": text_residual_ratio,
            "background_damage_ratio": background_damage_ratio,
            "edge_halo_score": edge_halo_score,
            "inpaint_runtime_ms": int((time.perf_counter() - started_at) * 1000),
            "peak_rss_mb": peak_rss_mb,
            "mask_coverage_ratio": mask_coverage,
            "candidate_mask_pixels": int(mask_stats.get("raw_candidate_pixels") or 0),
            "final_mask_pixels": int(mask_stats.get("refined_mask_pixels") or mask_pixels),
            "unmasked_candidate_ratio": unmasked_candidate_ratio,
            "final_to_candidate_ratio": final_to_candidate_ratio,
            "mask_source": mask_stats.get("mask_source"),
            "mask_refine_mode": mask_stats.get("mode"),
            "quality_score": score,
        },
        "warnings": warnings,
    }


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None
