from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def detect_apple_vision(
    *,
    image_path: Path,
    min_confidence: float,
    max_regions: int,
) -> dict[str, Any]:
    try:
        import Foundation
        import Quartz
        import Vision
    except Exception as error:
        raise RuntimeError("PyObjC Vision and Quartz are required for apple-vision provider") from error

    if not image_path.exists():
        raise FileNotFoundError(f"input image not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to load image: {image_path}")
    height, width = image.shape[:2]

    url = Foundation.NSURL.fileURLWithPath_(str(image_path))
    source = Quartz.CGImageSourceCreateWithURL(url, None)
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    if cg_image is None:
        raise ValueError(f"failed to create CGImage for: {image_path}")

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    try:
        request.setRecognitionLanguages_(["ja-JP", "zh-Hans", "zh-Hant", "en-US", "ko-KR"])
    except Exception:
        pass
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Apple Vision request failed: {error}")

    observations = list(request.results() or [])
    items = []
    for observation in observations:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        candidate = candidates[0]
        confidence = float(candidate.confidence())
        if confidence < min_confidence:
            continue
        bbox = observation.boundingBox()
        normalized = _vision_bbox_to_top_left(
            float(bbox.origin.x),
            float(bbox.origin.y),
            float(bbox.size.width),
            float(bbox.size.height),
        )
        items.append(
            {
                "bbox": normalized,
                "confidence": round(confidence, 4),
                "text": str(candidate.string()),
            }
        )
    items = sorted(items, key=lambda item: (item["bbox"][1], item["bbox"][0]))[:max_regions]
    textlines = []
    for index, item in enumerate(items, 1):
        textlines.append(
            {
                "id": f"line_{index:03d}",
                "bbox": item["bbox"],
                "confidence": item["confidence"],
                "kind": "text_line",
                "reading_order": index,
                "polygon": _bbox_to_polygon(item["bbox"]),
                "metadata": {
                    "source": "apple_vision",
                    "recognized_text": item["text"],
                    "license": "Apple platform API",
                },
            }
        )
    regions = []
    for index, item in enumerate(items, 1):
        regions.append(
            {
                "id": f"reg_{index:03d}",
                "bbox": item["bbox"],
                "confidence": item["confidence"],
                "kind": "text_region",
                "reading_order": index,
                "polygon": _bbox_to_polygon(item["bbox"]),
                "metadata": {
                    "source": "apple_vision",
                    "recognized_text": item["text"],
                    "license": "Apple platform API",
                },
            }
        )
    raw_mask = _mask_from_boxes(items, width=width, height=height)
    return {
        "schema_version": "apple_vision_adapter.v1",
        "provider": "apple-vision",
        "regions": regions,
        "textlines": textlines,
        "raw_mask": raw_mask,
        "quality": {
            "candidate_count": len(items),
            "recognition_level": "accurate",
            "min_confidence": min_confidence,
        },
    }


def _vision_bbox_to_top_left(x: float, y: float, w: float, h: float) -> list[float]:
    x1 = max(0.0, min(1.0, x))
    y1 = max(0.0, min(1.0, 1.0 - y - h))
    x2 = max(x1, min(1.0, x1 + max(0.0, w)))
    y2 = max(y1, min(1.0, y1 + max(0.0, h)))
    return [round(x1, 6), round(y1, 6), round(x2 - x1, 6), round(y2 - y1, 6)]


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _mask_from_boxes(items: list[dict[str, Any]], *, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in items:
        x, y, w, h = item["bbox"]
        x1 = int(round(x * width))
        y1 = int(round(y * height))
        x2 = int(round((x + w) * width))
        y2 = int(round((y + h) * height))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask
