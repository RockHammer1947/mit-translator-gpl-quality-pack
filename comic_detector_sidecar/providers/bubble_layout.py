from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from comic_detector_sidecar.contracts.documents import BBox, BubbleRegion, ComicTextRegion, LayoutRegion


def build_bubble_and_layout_regions(
    image_path: Path,
    textlines: list[ComicTextRegion],
    raw_mask: np.ndarray | None,
    *,
    provider: str,
) -> tuple[list[BubbleRegion], list[LayoutRegion], np.ndarray | None, np.ndarray | None]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return [], _layout_regions_from_textlines(textlines, provider=provider), raw_mask, None
    height, width = image.shape[:2]
    groups = _cluster_textlines(textlines)
    bubble_mask = np.zeros((height, width), dtype=np.uint8)
    refined_mask = np.zeros((height, width), dtype=np.uint8)
    if raw_mask is not None:
        refined_mask = cv2.resize(raw_mask, (width, height)) if raw_mask.shape[:2] != (height, width) else raw_mask.copy()
        refined_mask[refined_mask > 0] = 255

    bubbles: list[BubbleRegion] = []
    layout_regions: list[LayoutRegion] = []
    for index, group in enumerate(groups, 1):
        union = _union_bbox([line.bbox.root for line in group])
        pixel_rect = _bbox_to_xyxy(_pad_bbox(union, 0.018), width, height)
        mask, rect = extract_balloon_region(image, pixel_rect, enlarge_ratio=1.6)
        if mask is not None and rect is not None:
            x1, y1, x2, y2 = rect
            bubble_mask[y1:y2, x1:x2] = np.maximum(bubble_mask[y1:y2, x1:x2], mask)
            bubble_bbox = _xyxy_to_bbox([x1, y1, x2, y2], width, height)
            confidence = min(0.98, max(line.confidence for line in group) + 0.05)
            background_type = _background_type(image[y1:y2, x1:x2], mask)
            source_ids = [line.id for line in group]
            bubble = BubbleRegion(
                id=f"bubble_{index:03d}",
                bbox=BBox(bubble_bbox),
                polygon=_bbox_to_polygon(bubble_bbox),
                confidence=confidence,
                source_textline_ids=source_ids,
                background_type=background_type,
                metadata={
                    "source": "mit_balloon_extractor",
                    "fallback": False,
                    "textline_bbox": union,
                },
            )
        else:
            bubble_bbox = _pad_bbox(union, 0.025)
            source_ids = [line.id for line in group]
            bubble = BubbleRegion(
                id=f"bubble_{index:03d}",
                bbox=BBox(bubble_bbox),
                polygon=_bbox_to_polygon(bubble_bbox),
                confidence=max(line.confidence for line in group) * 0.75,
                source_textline_ids=source_ids,
                background_type="unknown",
                metadata={"source": "textline_union_fallback", "fallback": True},
            )
        bubbles.append(bubble)
        layout_regions.append(
            LayoutRegion(
                id=f"layout_{index:03d}",
                bbox=bubble.bbox,
                polygon=bubble.polygon,
                kind=_layout_kind(group, bubble.background_type),
                confidence=bubble.confidence,
                source_textline_ids=bubble.source_textline_ids,
                source_bubble_id=bubble.id,
                render_priority=index,
                metadata={
                    "source": "bubble_layout_region",
                    "textline_count": len(group),
                    "textline_bbox": union,
                },
            )
        )
        for line in group:
            x1, y1, x2, y2 = _bbox_to_xyxy(line.bbox.root, width, height)
            cv2.rectangle(refined_mask, (x1, y1), (x2, y2), 255, -1)

    if not layout_regions:
        layout_regions = _layout_regions_from_textlines(textlines, provider=provider)
    if refined_mask.any():
        refined_mask = cv2.dilate(refined_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    else:
        refined_mask = raw_mask
    return bubbles, layout_regions, refined_mask, bubble_mask if bubble_mask.any() else None


def extract_balloon_region(
    image: np.ndarray,
    rect: list[int],
    *,
    enlarge_ratio: float = 1.0,
) -> tuple[np.ndarray | None, list[int] | None]:
    # GPL-derived from MIT rendering/ballon_extractor.py; reduced to headless mask extraction.
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = rect
    if enlarge_ratio > 1:
        x1, y1, x2, y2 = _enlarge_window([x1, y1, x2, y2], image_width, image_height, enlarge_ratio)
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = image[y1:y2, x1:x2].copy()
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return None, None
    scale = 1.0
    if h > 300 and w > 300:
        scale = 0.6
    elif h < 120 or w < 120:
        scale = 1.4
    original = crop
    if scale != 1.0:
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = crop.shape[:2]
    area = h * w
    blurred = cv2.GaussianBlur(crop, (3, 3), cv2.BORDER_DEFAULT)
    edges = cv2.Canny(blurred, 70, 140, L2gradient=True, apertureSize=3)
    cv2.rectangle(edges, (0, 0), (w - 1, h - 1), 255, 1, cv2.LINE_8)
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.rectangle(edges, (0, 0), (w - 1, h - 1), 0, 1, cv2.LINE_8)
    balloon_mask = np.zeros((h, w), np.uint8)
    best_area = np.inf
    contour_mask = np.zeros((h, w), np.uint8)
    seed = (int(w / 2), int(h / 2))
    for contour in contours:
        contour_rect = cv2.boundingRect(contour)
        if contour_rect[2] * contour_rect[3] < area * 0.35:
            continue
        contour_mask = cv2.drawContours(contour_mask, [contour], -1, 255, 2)
        candidate = contour_mask.copy()
        cv2.rectangle(contour_mask, (0, 0), (w - 1, h - 1), 255, 1, cv2.LINE_8)
        filled_area, _, _, _ = cv2.floodFill(candidate, mask=None, seedPoint=seed, flags=4, newVal=127, loDiff=(10, 10, 10), upDiff=(10, 10, 10))
        if area * 0.25 < filled_area < best_area:
            best_area = filled_area
            balloon_mask = candidate
    if not np.isfinite(best_area):
        return None, None
    balloon_mask = 127 - balloon_mask
    balloon_mask = cv2.dilate(balloon_mask, np.ones((3, 3), np.uint8), iterations=1)
    flood_area, _, _, _ = cv2.floodFill(balloon_mask, mask=None, seedPoint=seed, flags=4, newVal=30, loDiff=(10, 10, 10), upDiff=(10, 10, 10))
    balloon_mask = 30 - balloon_mask
    _, balloon_mask = cv2.threshold(balloon_mask, 1, 255, cv2.THRESH_BINARY)
    balloon_mask = cv2.bitwise_not(balloon_mask, balloon_mask)
    kernel_size = int(np.sqrt(max(1, flood_area)) / 30)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        balloon_mask = cv2.dilate(balloon_mask, kernel, iterations=1)
        balloon_mask = cv2.erode(balloon_mask, kernel, iterations=1)
    if scale != 1.0:
        balloon_mask = cv2.resize(balloon_mask, (original.shape[1], original.shape[0]))
    return balloon_mask, [x1, y1, x2, y2]


def _cluster_textlines(textlines: list[ComicTextRegion]) -> list[list[ComicTextRegion]]:
    ordered = sorted(textlines, key=lambda line: (line.reading_order, line.bbox.root[1], line.bbox.root[0]))
    groups: list[list[ComicTextRegion]] = []
    for line in ordered:
        best_index = None
        best_distance = 999.0
        for index, group in enumerate(groups):
            union = _union_bbox([item.bbox.root for item in group])
            distance = _bbox_gap(union, line.bbox.root)
            same_direction = _same_orientation(group[0].bbox.root, line.bbox.root)
            if distance < best_distance and same_direction and distance < 0.055:
                best_index = index
                best_distance = distance
        if best_index is None:
            groups.append([line])
        else:
            groups[best_index].append(line)
    return groups


def _layout_regions_from_textlines(textlines: list[ComicTextRegion], *, provider: str) -> list[LayoutRegion]:
    return [
        LayoutRegion(
            id=f"layout_{index:03d}",
            bbox=line.bbox,
            polygon=line.polygon,
            kind="text_cluster",
            confidence=line.confidence,
            source_textline_ids=[line.id],
            render_priority=index,
            metadata={"source": "textline_layout_fallback", "provider": provider},
        )
        for index, line in enumerate(textlines, 1)
    ]


def _enlarge_window(rect: list[int], image_width: int, image_height: int, ratio: float) -> list[int]:
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return [0, 0, 0, 0]
    aspect = h / max(1, w)
    roots = np.roots([aspect, w + h * aspect, (1 - ratio) * w * h])
    roots.sort()
    delta = int(round(float(roots[-1]) / 2))
    delta_w = min(x1, image_width - x2, int(delta * aspect))
    delta_h = min(y1, image_height - y2, delta)
    return [
        int(np.clip(x1 - delta_w, 0, image_width - 1)),
        int(np.clip(y1 - delta_h, 0, image_height - 1)),
        int(np.clip(x2 + delta_w, 0, image_width - 1)),
        int(np.clip(y2 + delta_h, 0, image_height - 1)),
    ]


def _background_type(crop: np.ndarray, mask: np.ndarray | None) -> str:
    if crop.size == 0:
        return "unknown"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if mask is not None and mask.shape[:2] == gray.shape:
        pixels = gray[mask > 0]
    else:
        pixels = gray.reshape(-1)
    if pixels.size == 0:
        return "unknown"
    mean = float(np.mean(pixels))
    if mean > 180:
        return "white"
    if mean < 80:
        return "black"
    return "unknown"


def _layout_kind(group: list[ComicTextRegion], background_type: str) -> str:
    union = _union_bbox([line.bbox.root for line in group])
    if background_type in {"white", "black"} and union[2] * union[3] > 0.001:
        return "bubble"
    if len(group) == 1 and (union[2] > union[3] * 2.2 or union[3] > union[2] * 2.2):
        return "sfx"
    return "text_cluster"


def _same_orientation(left: list[float], right: list[float]) -> bool:
    left_vertical = left[3] > left[2] * 1.25
    right_vertical = right[3] > right[2] * 1.25
    return left_vertical == right_vertical


def _bbox_gap(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    dx = max(rx - (lx + lw), lx - (rx + rw), 0.0)
    dy = max(ry - (ly + lh), ly - (ry + rh), 0.0)
    return float(np.hypot(dx, dy))


def _union_bbox(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return [0, 0, 0, 0]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return _clamp_bbox([x1, y1, x2 - x1, y2 - y1])


def _pad_bbox(bbox: list[float], pad: float) -> list[float]:
    x, y, w, h = bbox
    return _clamp_bbox([x - pad, y - pad, w + pad * 2, h + pad * 2])


def _clamp_bbox(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    x = max(0.0, min(1.0, float(x)))
    y = max(0.0, min(1.0, float(y)))
    w = max(0.0, min(1.0 - x, float(w)))
    h = max(0.0, min(1.0 - y, float(h)))
    return [round(x, 6), round(y, 6), round(w, 6), round(h, 6)]


def _bbox_to_xyxy(bbox: list[float], width: int, height: int) -> list[int]:
    x, y, w, h = bbox
    return [
        int(max(0, min(width - 1, round(x * width)))),
        int(max(0, min(height - 1, round(y * height)))),
        int(max(0, min(width - 1, round((x + w) * width)))),
        int(max(0, min(height - 1, round((y + h) * height)))),
    ]


def _xyxy_to_bbox(rect: list[int], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = rect
    return _clamp_bbox([x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height])


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
