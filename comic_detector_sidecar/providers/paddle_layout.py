from __future__ import annotations

from pathlib import Path

from comic_detector_sidecar.adapters.paddle_layout_adapter import detect_paddle_layout
from comic_detector_sidecar.contracts.documents import BBox, ComicTextRegion, DetectorOptions, LayoutRegion
from comic_detector_sidecar.model_manager import status_for_paddle_layout
from comic_detector_sidecar.providers.base import ComicDetectionResult, ComicDetectorProvider


class PaddleLayoutComicDetectorProvider(ComicDetectorProvider):
    name = "paddle-layout"

    def is_available(self, options: DetectorOptions) -> bool:
        return status_for_paddle_layout().available

    def doctor(self, options: DetectorOptions) -> dict:
        payload = status_for_paddle_layout().to_doctor_payload()
        payload["name"] = self.name
        payload["backend"] = "paddlex-pp-doclayoutv3"
        return payload

    def detect_image(self, image_path: Path, options: DetectorOptions) -> ComicDetectionResult:
        payload = detect_paddle_layout(
            image_path=image_path,
            score_threshold=_score_threshold(options),
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
                metadata={"source": "paddle_layout_text_box", "provider": self.name},
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


def _score_threshold(options: DetectorOptions) -> float:
    if options.coverage_preset == "quality":
        return min(options.text_threshold, 0.35)
    if options.coverage_preset == "fast":
        return max(options.text_threshold, 0.6)
    return min(options.text_threshold, 0.45)


def _parse_region(item: dict, default_kind: str) -> ComicTextRegion:
    bbox = [float(value) for value in item["bbox"]]
    return ComicTextRegion(
        id=str(item["id"]),
        bbox=BBox(bbox),
        confidence=float(item.get("confidence", 0.0)),
        kind=item.get("kind") or default_kind,
        reading_order=int(item.get("reading_order", 1)),
        provider="paddle-layout",
        polygon=item.get("polygon") or _bbox_to_polygon(bbox),
        source_region_id=item.get("source_region_id"),
        metadata=item.get("metadata") or {},
    )


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
