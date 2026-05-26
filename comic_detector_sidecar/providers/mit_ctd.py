from __future__ import annotations

from pathlib import Path

from comic_detector_sidecar.adapters.mit_ctd_adapter import detect_mit_ctd
from comic_detector_sidecar.contracts.documents import BBox, ComicTextRegion, DetectorOptions
from comic_detector_sidecar.model_manager import default_comic_text_detector_model_path, status_for_comic_text_detector
from comic_detector_sidecar.providers.base import ComicDetectionResult, ComicDetectorProvider
from comic_detector_sidecar.providers.bubble_layout import build_bubble_and_layout_regions


class MitCtdComicDetectorProvider(ComicDetectorProvider):
    name = "mit-ctd"

    def is_available(self, options: DetectorOptions) -> bool:
        return status_for_comic_text_detector(options.model_path, None).available

    def doctor(self, options: DetectorOptions) -> dict:
        payload = status_for_comic_text_detector(options.model_path, None).to_doctor_payload()
        payload["name"] = self.name
        payload["license"] = "GPL-3.0 MIT-derived provider"
        payload["backend"] = "opencv-dnn-onnx"
        payload["coverage_preset"] = options.coverage_preset
        return payload

    def detect_image(self, image_path: Path, options: DetectorOptions) -> ComicDetectionResult:
        model_path = options.model_path or default_comic_text_detector_model_path()
        tuned = _apply_preset(options)
        payload = detect_mit_ctd(
            image_path=image_path,
            model_path=model_path,
            detection_size=tuned.detection_size,
            text_threshold=tuned.text_threshold,
            box_threshold=tuned.box_threshold,
            unclip_ratio=tuned.unclip_ratio,
            nms_threshold=tuned.nms_threshold,
            min_textline_area_ratio=tuned.min_textline_area_ratio,
            max_regions=tuned.max_regions,
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
                "coverage_preset": tuned.coverage_preset,
                "textline_count": len(textlines),
                "region_count": len(regions),
                "bubble_count": len(bubbles),
                "layout_region_count": len(layout_regions),
                "effective_detection_size": payload.get("effective_detection_size"),
                "requested_detection_size": payload.get("requested_detection_size"),
            },
        )


def _apply_preset(options: DetectorOptions) -> DetectorOptions:
    data = options.model_dump()
    if options.coverage_preset == "fast":
        data.update({"detection_size": 1024, "text_threshold": 0.45, "box_threshold": 0.45, "unclip_ratio": 1.5})
    elif options.coverage_preset == "quality":
        data.update({"detection_size": 1536, "text_threshold": 0.28, "box_threshold": 0.28, "unclip_ratio": 2.0})
    else:
        data.update({"detection_size": 1536, "text_threshold": 0.35, "box_threshold": 0.35, "unclip_ratio": 1.8})
    return DetectorOptions(**data)


def _parse_region(item: dict, default_kind: str) -> ComicTextRegion:
    bbox = [float(value) for value in item["bbox"]]
    return ComicTextRegion(
        id=str(item["id"]),
        bbox=BBox(bbox),
        confidence=float(item.get("confidence", 0.0)),
        kind=item.get("kind") or default_kind,
        reading_order=int(item.get("reading_order", 1)),
        provider="mit-ctd",
        polygon=item.get("polygon") or _bbox_to_polygon(bbox),
        source_region_id=item.get("source_region_id"),
        metadata=item.get("metadata") or {},
    )


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
