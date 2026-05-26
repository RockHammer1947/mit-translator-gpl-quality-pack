import json
import uuid
from pathlib import Path

import numpy as np

from comic_detector_sidecar import __version__
from comic_detector_sidecar.contracts.documents import (
    BubbleRegionsDocument,
    ComicDetectorManifest,
    ComicTextDetectionDocument,
    LayoutRegionsDocument,
    DetectorOptions,
    TEXTLINES_SCHEMA_VERSION,
)
from comic_detector_sidecar.providers.factory import get_provider
from comic_detector_sidecar.utils.jsonl import emit_event
from comic_detector_sidecar.utils.paths import ensure_input_image


def detect_image(
    input_path: Path,
    provider_name: str,
    options: DetectorOptions,
    output_dir: Path,
    manifest_path: Path | None,
    job_id: str | None,
    jsonl: bool,
) -> ComicTextDetectionDocument:
    resolved_job_id = job_id or f"comic_detect_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    regions_path = output_dir / "comic_text_regions.json"
    textlines_path = output_dir / "textlines.json"
    bubbles_path = output_dir / "bubble_regions.json"
    layout_regions_path = output_dir / "layout_regions.json"
    raw_mask_path = output_dir / "text_mask_raw.png"
    refined_mask_path = output_dir / "text_mask_refined.png"
    bubble_mask_path = output_dir / "bubble_mask.png"
    detector_quality_path = output_dir / "detector_quality_report.json"
    debug_overlay_path = output_dir / "detector_debug_overlay.png"
    resolved_manifest_path = manifest_path or output_dir / "comic_detector_manifest.json"
    warnings: list[dict[str, str]] = []

    try:
        ensure_input_image(input_path)
        provider = get_provider(provider_name)
        emit_event(
            {
                "type": "started",
                "job_id": resolved_job_id,
                "sidecar_version": __version__,
                "provider": provider.name,
            },
            jsonl,
        )
        detection = provider.detect_image(input_path, options)
        if detection.raw_mask is not None and detection.refined_mask is None:
            detection.refined_mask = detection.raw_mask
        if detection.raw_mask is not None and detection.bubble_mask is None:
            detection.bubble_mask = np.zeros_like(detection.raw_mask)
        regions = [region for region in detection.regions if region.confidence >= options.min_confidence]
        textlines = [line for line in detection.textlines if line.confidence >= options.min_confidence]
        bubbles = detection.bubbles or []
        layout_regions = detection.layout_regions or []
        for region in regions:
            emit_event(
                {
                    "type": "region_detected",
                    "job_id": resolved_job_id,
                    "region_id": region.id,
                    "bbox": region.bbox.root,
                    "confidence": region.confidence,
                    "kind": region.kind,
                    "polygon": region.polygon,
                },
                jsonl,
            )
        for line in textlines:
            emit_event(
                {
                    "type": "textline_detected",
                    "job_id": resolved_job_id,
                    "textline_id": line.id,
                    "bbox": line.bbox.root,
                    "confidence": line.confidence,
                    "polygon": line.polygon,
                },
                jsonl,
            )

        doc = ComicTextDetectionDocument(
            job_id=resolved_job_id,
            provider=provider.name,
            source={"type": "image", "path": str(input_path)},
            regions=regions,
        )
        textlines_doc = ComicTextDetectionDocument(
            schema_version=TEXTLINES_SCHEMA_VERSION,
            job_id=resolved_job_id,
            provider=provider.name,
            source={"type": "image", "path": str(input_path)},
            regions=textlines,
        )
        bubbles_doc = BubbleRegionsDocument(
            job_id=resolved_job_id,
            provider=provider.name,
            source={"type": "image", "path": str(input_path)},
            regions=bubbles,
        )
        layout_regions_doc = LayoutRegionsDocument(
            job_id=resolved_job_id,
            provider=provider.name,
            source={"type": "image", "path": str(input_path)},
            regions=layout_regions,
        )
        _write_json(regions_path, doc.model_dump(mode="json"))
        _write_json(textlines_path, textlines_doc.model_dump(mode="json"))
        _write_json(bubbles_path, bubbles_doc.model_dump(mode="json"))
        _write_json(layout_regions_path, layout_regions_doc.model_dump(mode="json"))
        emit_event(
            {"type": "artifact", "job_id": resolved_job_id, "kind": "comic_text_regions", "path": str(regions_path)},
            jsonl,
        )
        emit_event(
            {"type": "artifact", "job_id": resolved_job_id, "kind": "comic_textlines", "path": str(textlines_path)},
            jsonl,
        )
        emit_event(
            {"type": "artifact", "job_id": resolved_job_id, "kind": "bubble_regions", "path": str(bubbles_path)},
            jsonl,
        )
        emit_event(
            {"type": "artifact", "job_id": resolved_job_id, "kind": "layout_regions", "path": str(layout_regions_path)},
            jsonl,
        )
        raw_mask_coverage_ratio = 0.0
        if detection.raw_mask is not None:
            raw_mask_coverage_ratio = _write_mask(raw_mask_path, detection.raw_mask)
            emit_event(
                {"type": "artifact", "job_id": resolved_job_id, "kind": "text_mask_raw", "path": str(raw_mask_path)},
                jsonl,
            )
        refined_mask_coverage_ratio = 0.0
        if detection.refined_mask is not None:
            refined_mask_coverage_ratio = _write_mask(refined_mask_path, detection.refined_mask)
            emit_event(
                {"type": "artifact", "job_id": resolved_job_id, "kind": "text_mask_refined", "path": str(refined_mask_path)},
                jsonl,
            )
        bubble_mask_coverage_ratio = 0.0
        if detection.bubble_mask is not None:
            bubble_mask_coverage_ratio = _write_mask(bubble_mask_path, detection.bubble_mask)
            emit_event(
                {"type": "artifact", "job_id": resolved_job_id, "kind": "bubble_mask", "path": str(bubble_mask_path)},
                jsonl,
            )
        overlay_mask = detection.refined_mask if detection.refined_mask is not None else detection.raw_mask
        if overlay_mask is not None:
            _write_debug_overlay(input_path, debug_overlay_path, regions, textlines, overlay_mask, layout_regions=layout_regions)
            emit_event(
                {
                    "type": "artifact",
                    "job_id": resolved_job_id,
                    "kind": "detector_debug_overlay",
                    "path": str(debug_overlay_path),
                },
                jsonl,
            )

        if not regions:
            warnings.append({"code": "EMPTY_TEXT_REGION_RESULT", "message": "No comic text regions detected."})
            emit_event(
                {
                    "type": "warning",
                    "job_id": resolved_job_id,
                    "code": "EMPTY_TEXT_REGION_RESULT",
                    "message": "No comic text regions detected.",
                },
                jsonl,
            )

        detector_quality = {
            "schema_version": "detector_quality_report.v1",
            "job_id": resolved_job_id,
            "provider": provider.name,
            "source": {"type": "image", "path": str(input_path)},
            "summary": {
                "region_count": len(regions),
                "textline_count": len(textlines),
                "bubble_count": len(bubbles),
                "layout_region_count": len(layout_regions),
                "raw_mask_coverage_ratio": raw_mask_coverage_ratio,
                "refined_mask_coverage_ratio": refined_mask_coverage_ratio,
                "bubble_mask_coverage_ratio": bubble_mask_coverage_ratio,
            },
            "provider_quality": detection.quality or {},
            "warnings": warnings,
        }
        _write_json(detector_quality_path, detector_quality)
        emit_event(
            {"type": "artifact", "job_id": resolved_job_id, "kind": "detector_quality_report", "path": str(detector_quality_path)},
            jsonl,
        )

        manifest = ComicDetectorManifest(
            job_id=resolved_job_id,
            status="completed",
            provider=provider.name,
            source={"type": "image", "path": str(input_path)},
            artifacts={
                "regions": str(regions_path),
                "textlines": str(textlines_path),
                "bubbles": str(bubbles_path),
                "layout_regions": str(layout_regions_path),
                "raw_mask": str(raw_mask_path) if detection.raw_mask is not None else "",
                "refined_mask": str(refined_mask_path) if detection.refined_mask is not None else "",
                "bubble_mask": str(bubble_mask_path) if detection.bubble_mask is not None else "",
                "debug_overlay": str(debug_overlay_path) if detection.raw_mask is not None else "",
                "detector_quality_report": str(detector_quality_path),
                "manifest": str(resolved_manifest_path),
            },
            summary={
                "region_count": len(regions),
                "textline_count": len(textlines),
                "bubble_count": len(bubbles),
                "layout_region_count": len(layout_regions),
                "raw_mask_coverage_ratio": raw_mask_coverage_ratio,
                "refined_mask_coverage_ratio": refined_mask_coverage_ratio,
                "bubble_mask_coverage_ratio": bubble_mask_coverage_ratio,
            },
            warnings=warnings,
        )
        _write_json(resolved_manifest_path, manifest.model_dump(mode="json"))
        emit_event(
            {
                "type": "done",
                "job_id": resolved_job_id,
                "status": "completed",
                "manifest_path": str(resolved_manifest_path),
            },
            jsonl,
        )
        if not jsonl:
            print(json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")))
        return doc
    except Exception as error:
        emit_event(
            {
                "type": "error",
                "job_id": resolved_job_id,
                "code": _error_code(error),
                "message": str(error),
            },
            jsonl,
        )
        if not jsonl:
            print(json.dumps({"error": {"code": _error_code(error), "message": str(error)}}, ensure_ascii=False))
        raise


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_mask(path: Path, mask) -> float:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)
    pixels = mask.shape[0] * mask.shape[1] if mask is not None else 0
    return float(np.count_nonzero(mask) / pixels) if pixels else 0.0


def _write_debug_overlay(input_path: Path, output_path: Path, regions, textlines, mask, layout_regions=None) -> None:
    import cv2

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    height, width = image.shape[:2]
    if mask is not None:
        mask_resized = cv2.resize(mask, (width, height)) if mask.shape[:2] != (height, width) else mask
        overlay = image.copy()
        overlay[mask_resized > 0] = (0, 180, 255)
        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
    for region in regions:
        _draw_bbox(image, region.bbox.root, width, height, (255, 80, 20), 2)
    for region in layout_regions or []:
        _draw_bbox(image, region.bbox.root, width, height, (180, 40, 255), 2)
    for line in textlines:
        _draw_bbox(image, line.bbox.root, width, height, (20, 220, 80), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def _draw_bbox(image, bbox, width: int, height: int, color, thickness: int) -> None:
    import cv2

    x, y, w, h = bbox
    left = int(max(0.0, min(1.0, x)) * width)
    top = int(max(0.0, min(1.0, y)) * height)
    right = int(max(0.0, min(1.0, x + w)) * width)
    bottom = int(max(0.0, min(1.0, y + h)) * height)
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)


def _error_code(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        if "adapter" in str(error) or "comic-text-detector" in str(error):
            return "DETECTOR_BINARY_NOT_FOUND"
        return "INPUT_NOT_FOUND"
    if isinstance(error, NotImplementedError):
        return "DETECTOR_PROVIDER_NOT_IMPLEMENTED"
    if "not supported" in str(error):
        return "DETECTOR_PROVIDER_NOT_FOUND"
    return "DETECTOR_EXECUTION_FAILED"
