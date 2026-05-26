from __future__ import annotations

from pathlib import Path

from comic_detector_sidecar.adapters.mmocr_adapter import DEFAULT_MMOCR_TEXTDET_MODEL, detect_mmocr
from comic_detector_sidecar.contracts.documents import BBox, ComicTextRegion, DetectorOptions
from comic_detector_sidecar.model_manager import status_for_mmocr
from comic_detector_sidecar.providers.base import ComicDetectionResult, ComicDetectorProvider
from comic_detector_sidecar.providers.bubble_layout import build_bubble_and_layout_regions


class MmocrComicDetectorProvider(ComicDetectorProvider):
    name = "mmocr"

    def is_available(self, options: DetectorOptions) -> bool:
        return status_for_mmocr().available

    def doctor(self, options: DetectorOptions) -> dict:
        payload = status_for_mmocr().to_doctor_payload()
        payload["name"] = self.name
        payload["backend"] = "openmmlab-mmocr-textdet"
        payload["default_model"] = DEFAULT_MMOCR_TEXTDET_MODEL
        payload["coverage_preset"] = options.coverage_preset
        return payload

    def detect_image(self, image_path: Path, options: DetectorOptions) -> ComicDetectionResult:
        payload = detect_mmocr(
            image_path=image_path,
            model_name=DEFAULT_MMOCR_TEXTDET_MODEL,
            score_threshold=_score_threshold(options),
            max_regions=options.max_regions,
        )
        regions = [_parse_region(item, "text_region") for item in payload["regions"]]
        textlines = [_parse_region(item, "text_line") for item in payload["textlines"]]
        raw_mask = payload.get("raw_mask")
        bubbles, layout_regions, refined_mask, bubble_mask = build_bubble_and_layout_regions(
            image_path,
            textlines,
            raw_mask,
            provider=self.name,
        )
        return ComicDetectionResult(
            regions=regions,
            textlines=textlines,
            raw_mask=raw_mask,
            refined_mask=refined_mask,
            bubble_mask=bubble_mask,
            bubbles=bubbles,
            layout_regions=layout_regions,
            quality={
                "provider": self.name,
                "coverage_preset": options.coverage_preset,
                "bubble_count": len(bubbles),
                "layout_region_count": len(layout_regions),
                **(payload.get("quality") or {}),
            },
        )


def _score_threshold(options: DetectorOptions) -> float:
    if options.coverage_preset == "quality":
        return min(options.text_threshold, 0.25)
    if options.coverage_preset == "fast":
        return max(options.text_threshold, 0.5)
    return min(options.text_threshold, 0.35)


def _parse_region(item: dict, default_kind: str) -> ComicTextRegion:
    bbox = [float(value) for value in item["bbox"]]
    return ComicTextRegion(
        id=str(item["id"]),
        bbox=BBox(bbox),
        confidence=float(item.get("confidence", 0.0)),
        kind=item.get("kind") or default_kind,
        reading_order=int(item.get("reading_order", 1)),
        provider="mmocr",
        polygon=item.get("polygon") or _bbox_to_polygon(bbox),
        source_region_id=item.get("source_region_id"),
        metadata=item.get("metadata") or {},
    )


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
