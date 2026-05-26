from pathlib import Path

import cv2
import numpy as np

from comic_detector_sidecar.contracts.documents import BBox, ComicTextRegion, DetectorOptions
from comic_detector_sidecar.providers.base import ComicDetectionResult, ComicDetectorProvider


class HeuristicComicDetectorProvider(ComicDetectorProvider):
    name = "heuristic"

    def is_available(self, options: DetectorOptions) -> bool:
        return True

    def doctor(self, options: DetectorOptions) -> dict:
        return {"name": self.name, "available": True}

    def detect_image(self, image_path: Path, options: DetectorOptions) -> ComicDetectionResult:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        height, width = image.shape[:2]
        threshold = int(max(40, min(220, 255 * (1.0 - options.text_threshold))))
        raw_mask = (image < threshold).astype("uint8") * 255
        regions = _seed_layout_regions(width, height)
        textline_boxes = _connected_text_regions(raw_mask)
        if not textline_boxes:
            textline_boxes = regions[: options.max_regions]
        regions.extend(textline_boxes)
        return ComicDetectionResult(
            regions=_dedupe_regions(regions, provider=self.name, max_regions=options.max_regions, kind="text_region", prefix="reg"),
            textlines=_dedupe_regions(
                textline_boxes,
                provider=self.name,
                max_regions=max(options.max_regions * 3, options.max_regions),
                kind="text_line",
                prefix="line",
            ),
            raw_mask=raw_mask,
        )


def _seed_layout_regions(width: int, height: int) -> list[list[float]]:
    aspect = width / height if height else 1.0
    regions: list[list[float]] = [
        [0.03, 0.02, 0.25, 0.22],
        [0.28, 0.02, 0.22, 0.22],
        [0.72, 0.03, 0.25, 0.24],
        [0.79, 0.25, 0.18, 0.18],
        [0.03, 0.50, 0.24, 0.22],
        [0.32, 0.50, 0.18, 0.23],
        [0.03, 0.72, 0.22, 0.24],
        [0.25, 0.70, 0.18, 0.25],
        [0.72, 0.76, 0.24, 0.18],
    ]
    if aspect < 0.85:
        regions.extend(
            [
                [0.00, 0.00, 1.00, 0.25],
                [0.00, 0.24, 1.00, 0.28],
                [0.00, 0.50, 1.00, 0.26],
                [0.00, 0.72, 1.00, 0.28],
            ]
        )
    return regions


def _connected_text_regions(mask: np.ndarray) -> list[list[float]]:
    height, width = mask.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[list[float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 90 or w < 6 or h < 10:
            continue
        if w > width * 0.65 or h > height * 0.50:
            continue
        regions.append(_pad_bbox([x / width, y / height, w / width, h / height], 0.008))
    return regions


def _dedupe_regions(
    regions: list[list[float]],
    provider: str,
    max_regions: int,
    kind: str,
    prefix: str,
) -> list[ComicTextRegion]:
    ranked = sorted(regions, key=lambda bbox: (bbox[1], bbox[0], bbox[2] * bbox[3]))
    kept: list[list[float]] = []
    for region in ranked:
        cleaned = _clamp_bbox(region)
        if cleaned[2] <= 0.01 or cleaned[3] <= 0.01:
            continue
        if any(_bbox_iou(cleaned, existing) > 0.45 for existing in kept):
            continue
        kept.append(cleaned)
        if len(kept) >= max_regions:
            break
    return [
        ComicTextRegion(
            id=f"{prefix}_{index:03d}",
            bbox=BBox(region),
            confidence=0.55,
            kind=kind,
            reading_order=index,
            provider=provider,
            polygon=_bbox_to_polygon(region),
            metadata={"source": "heuristic_layout"},
        )
        for index, region in enumerate(kept, 1)
    ]


def _pad_bbox(bbox: list[float], pad: float) -> list[float]:
    x, y, w, h = bbox
    return [x - pad, y - pad, w + pad * 2, h + pad * 2]


def _clamp_bbox(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    return [x, y, w, h]


def _bbox_iou(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    ix1 = max(lx, rx)
    iy1 = max(ly, ry)
    ix2 = min(lx + lw, rx + rw)
    iy2 = min(ly + lh, ry + rh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = lw * lh + rw * rh - inter
    return inter / union if union > 0 else 0.0


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
