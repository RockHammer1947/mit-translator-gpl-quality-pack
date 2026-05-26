from __future__ import annotations

from pathlib import Path

from comic_detector_sidecar.adapters.ogkalu_rtdetr_adapter import detect_ogkalu_rtdetr
from comic_detector_sidecar.contracts.documents import BBox, BubbleRegion, ComicTextRegion, DetectorOptions, LayoutRegion
from comic_detector_sidecar.model_manager import default_ogkalu_rtdetr_model_path, status_for_ogkalu_rtdetr
from comic_detector_sidecar.providers.base import ComicDetectionResult, ComicDetectorProvider


class OgkaluRtdetrComicDetectorProvider(ComicDetectorProvider):
    name = "ogkalu-rtdetr"

    def is_available(self, options: DetectorOptions) -> bool:
        return status_for_ogkalu_rtdetr(options.model_path).available

    def doctor(self, options: DetectorOptions) -> dict:
        payload = status_for_ogkalu_rtdetr(options.model_path).to_doctor_payload()
        payload["name"] = self.name
        payload["backend"] = "onnxruntime-cpu"
        payload["labels"] = ["bubble", "text_bubble", "text_free"]
        return payload

    def detect_image(self, image_path: Path, options: DetectorOptions) -> ComicDetectionResult:
        payload = detect_ogkalu_rtdetr(
            image_path=image_path,
            model_path=options.model_path or default_ogkalu_rtdetr_model_path(),
            score_threshold=_score_threshold(options),
            nms_threshold=options.nms_threshold,
            max_regions=options.max_regions,
        )
        regions = [_parse_region(item, "text_region") for item in payload["regions"]]
        textlines = [_parse_region(item, "text_line") for item in payload["textlines"]]
        bubbles = [_parse_bubble(item) for item in payload["bubbles"]]
        layout_regions = _layout_regions_from_bubbles_and_textlines(bubbles, textlines, provider=self.name)
        return ComicDetectionResult(
            regions=regions,
            textlines=textlines,
            raw_mask=payload.get("raw_mask"),
            refined_mask=payload.get("raw_mask"),
            bubble_mask=payload.get("bubble_mask"),
            bubbles=bubbles,
            layout_regions=layout_regions,
            quality={
                "provider": self.name,
                "coverage_preset": options.coverage_preset,
                **(payload.get("quality") or {}),
            },
        )


def _score_threshold(options: DetectorOptions) -> float:
    if options.coverage_preset == "quality":
        return min(options.text_threshold, 0.18)
    if options.coverage_preset == "fast":
        return max(options.text_threshold, 0.5)
    return min(options.text_threshold, 0.25)


def _parse_region(item: dict, default_kind: str) -> ComicTextRegion:
    bbox = [float(value) for value in item["bbox"]]
    return ComicTextRegion(
        id=str(item["id"]),
        bbox=BBox(bbox),
        confidence=float(item.get("confidence", 0.0)),
        kind=item.get("kind") or default_kind,
        reading_order=int(item.get("reading_order", 1)),
        provider="ogkalu-rtdetr",
        polygon=item.get("polygon") or _bbox_to_polygon(bbox),
        source_region_id=item.get("source_region_id"),
        metadata=item.get("metadata") or {},
    )


def _parse_bubble(item: dict) -> BubbleRegion:
    bbox = [float(value) for value in item["bbox"]]
    return BubbleRegion(
        id=str(item["id"]),
        bbox=BBox(bbox),
        polygon=item.get("polygon") or _bbox_to_polygon(bbox),
        confidence=float(item.get("confidence", 0.0)),
        source_textline_ids=[str(value) for value in item.get("source_textline_ids", [])],
        background_type=item.get("background_type", "unknown"),
        metadata=item.get("metadata") or {},
    )


def _layout_regions_from_bubbles_and_textlines(
    bubbles: list[BubbleRegion],
    textlines: list[ComicTextRegion],
    *,
    provider: str,
) -> list[LayoutRegion]:
    regions: list[LayoutRegion] = []
    assigned: set[str] = set()
    for index, bubble in enumerate(bubbles, 1):
        source_ids = bubble.source_textline_ids
        assigned.update(source_ids)
        regions.append(
            LayoutRegion(
                id=f"layout_{index:03d}",
                bbox=bubble.bbox,
                polygon=bubble.polygon,
                kind="bubble",
                confidence=bubble.confidence,
                source_textline_ids=source_ids,
                source_bubble_id=bubble.id,
                render_priority=index,
                metadata={"source": "ogkalu_bubble", "provider": provider},
            )
        )
    next_index = len(regions) + 1
    for line in textlines:
        if line.id in assigned:
            continue
        regions.append(
            LayoutRegion(
                id=f"layout_{next_index:03d}",
                bbox=line.bbox,
                polygon=line.polygon,
                kind="sfx" if line.metadata.get("label") == "text_free" else "text_cluster",
                confidence=line.confidence,
                source_textline_ids=[line.id],
                render_priority=next_index,
                metadata={"source": "ogkalu_text_fallback", "provider": provider},
            )
        )
        next_index += 1
    return regions


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
