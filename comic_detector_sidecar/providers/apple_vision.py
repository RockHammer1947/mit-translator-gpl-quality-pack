from __future__ import annotations

from pathlib import Path

from comic_detector_sidecar.adapters.apple_vision_adapter import detect_apple_vision
from comic_detector_sidecar.contracts.documents import BBox, ComicTextRegion, DetectorOptions, LayoutRegion
from comic_detector_sidecar.providers.base import ComicDetectionResult, ComicDetectorProvider


class AppleVisionComicDetectorProvider(ComicDetectorProvider):
    name = "apple-vision"

    def is_available(self, options: DetectorOptions) -> bool:
        return _can_import("Vision") and _can_import("Quartz")

    def doctor(self, options: DetectorOptions) -> dict:
        available = self.is_available(options)
        return {
            "name": self.name,
            "available": available,
            "installed": available,
            "model_ready": available,
            "needs_download": False,
            "backend": "macos-vision-framework",
            "license": "Apple platform API",
            "reason": None if available else "PyObjC Vision/Quartz bridge is not installed",
        }

    def detect_image(self, image_path: Path, options: DetectorOptions) -> ComicDetectionResult:
        payload = detect_apple_vision(
            image_path=image_path,
            min_confidence=_score_threshold(options),
            max_regions=options.max_regions,
        )
        regions = [_parse_region(item, "text_region") for item in payload["regions"]]
        textlines = [_parse_region(item, "text_line") for item in payload["textlines"]]
        layout_regions = [
            LayoutRegion(
                id=f"layout_{index:03d}",
                bbox=line.bbox,
                polygon=line.polygon,
                kind="text_cluster",
                confidence=line.confidence,
                source_textline_ids=[line.id],
                render_priority=index,
                metadata={"source": "apple_vision_text_box", "provider": self.name},
            )
            for index, line in enumerate(textlines, 1)
        ]
        return ComicDetectionResult(
            regions=regions,
            textlines=textlines,
            raw_mask=payload.get("raw_mask"),
            refined_mask=payload.get("raw_mask"),
            bubble_mask=None,
            bubbles=[],
            layout_regions=layout_regions,
            quality={"provider": self.name, "coverage_preset": options.coverage_preset, **(payload.get("quality") or {})},
        )


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _score_threshold(options: DetectorOptions) -> float:
    if options.coverage_preset == "quality":
        return min(options.text_threshold, 0.1)
    if options.coverage_preset == "fast":
        return max(options.text_threshold, 0.4)
    return min(options.text_threshold, 0.2)


def _parse_region(item: dict, default_kind: str) -> ComicTextRegion:
    bbox = [float(value) for value in item["bbox"]]
    return ComicTextRegion(
        id=str(item["id"]),
        bbox=BBox(bbox),
        confidence=float(item.get("confidence", 0.0)),
        kind=item.get("kind") or default_kind,
        reading_order=int(item.get("reading_order", 1)),
        provider="apple-vision",
        polygon=item.get("polygon") or _bbox_to_polygon(bbox),
        source_region_id=item.get("source_region_id"),
        metadata=item.get("metadata") or {},
    )


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
