from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from manga_cleaner_sidecar.contracts import CleanerBlock, CleanerConfig


@dataclass
class RefinedMaskResult:
    mask: np.ndarray
    stats: dict[str, Any]


@dataclass
class TextPolygon:
    block_id: str
    polygon: np.ndarray
    font_size: float


def build_refined_mask(
    source_image: Path,
    raw_mask_image: Path | None,
    detector_refined_mask_image: Path | None,
    blocks: list[CleanerBlock],
    config: CleanerConfig,
) -> RefinedMaskResult:
    gray = cv2.imread(str(source_image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Input image not found or unreadable: {source_image}")
    height, width = gray.shape[:2]
    raw_mask = _load_mask(raw_mask_image, width, height)
    detector_refined_mask = _load_mask(detector_refined_mask_image, width, height)
    candidate, mask_source = _candidate_mask(gray, raw_mask, detector_refined_mask, config)
    raw_candidate_pixels = int(np.count_nonzero(candidate))

    if config.mask_refine_mode == "disabled":
        mask = np.zeros_like(candidate)
        return RefinedMaskResult(
            mask=mask,
            stats=_with_image_stats(_stats(config.mask_refine_mode, mask_source, raw_candidate_pixels, 0, 0, 0, 0, {}, []), width, height, 0),
        )

    text_polygons = _collect_text_polygons(blocks, width, height)
    if not text_polygons:
        refined = _post_dilate(candidate, config)
        refined_pixels = int(np.count_nonzero(refined))
        return RefinedMaskResult(
            mask=refined,
            stats=_with_image_stats(_stats(
                config.mask_refine_mode,
                mask_source,
                raw_candidate_pixels,
                refined_pixels,
                0,
                0,
                0,
                {},
                ["NO_TEXT_POLYGONS"],
            ), width, height, refined_pixels),
        )

    if config.mask_refine_mode in ("raw_mask", "threshold"):
        refined = _clip_candidate_to_polygons(candidate, text_polygons, config, width, height)
        refined = _post_dilate(refined, config)
        refined_pixels = int(np.count_nonzero(refined))
        return RefinedMaskResult(
            mask=refined,
            stats=_with_image_stats(_stats(config.mask_refine_mode, mask_source, raw_candidate_pixels, refined_pixels, 0, 0, 0, {}, []), width, height, refined_pixels),
        )

    if config.mask_refine_mode == "mit_fit_text":
        return _refine_mit_fit_text(candidate, text_polygons, config, width, height, mask_source, raw_candidate_pixels)

    refined = np.zeros_like(candidate)
    component_count = 0
    removed_count = 0
    reject_reasons: dict[str, int] = {}
    block_metrics: list[dict[str, Any]] = []

    for index, text_polygon in enumerate(text_polygons, 1):
        block_id = text_polygon.block_id
        polygon = text_polygon.polygon
        left, top, right, bottom = _expanded_rect_for_polygon(polygon, width, height, config.mask_dilation_offset)
        if right <= left or bottom <= top:
            continue
        local_candidate = candidate[top:bottom, left:right]
        local_safe = _polygon_mask_for_rect(polygon, (left, top, right, bottom), config.mask_dilation_offset)
        local_candidate = cv2.bitwise_and(local_candidate, local_safe)
        local_refined, local_stats = _filter_components(local_candidate, local_safe, config)
        component_count += local_stats["component_count"]
        removed_count += local_stats["removed_component_count"]
        _merge_counts(reject_reasons, local_stats["component_reject_reasons"])
        refined[top:bottom, left:right] = np.maximum(refined[top:bottom, left:right], local_refined)
        block_metrics.append({"block_id": block_id, "polygon_index": index, **local_stats})

    refined = _post_dilate(refined, config)
    refined_pixels = int(np.count_nonzero(refined))
    residual_candidate_pixels = max(0, raw_candidate_pixels - refined_pixels)
    stats = _stats(
        config.mask_refine_mode,
        mask_source,
        raw_candidate_pixels,
        refined_pixels,
        residual_candidate_pixels,
        component_count,
        removed_count,
        reject_reasons,
        [],
    )
    stats["block_metrics"] = block_metrics
    stats["text_polygon_count"] = len(text_polygons)
    stats["residual_candidate_ratio"] = residual_candidate_pixels / raw_candidate_pixels if raw_candidate_pixels else 0.0
    stats = _with_image_stats(stats, width, height, refined_pixels)
    return RefinedMaskResult(mask=refined, stats=stats)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def save_mask_debug_overlay(source_image: Path, mask: np.ndarray, output: Path) -> None:
    source = Image.open(source_image).convert("RGBA")
    mask_img = Image.fromarray(mask).convert("L")
    red = Image.new("RGBA", source.size, (255, 0, 0, 90))
    overlay = Image.composite(red, source, mask_img)
    blended = Image.blend(source, overlay, 0.45)
    ImageDraw.Draw(blended).text((8, 8), "manga cleaner mask debug", fill=(255, 0, 0, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    blended.convert("RGB").save(output)


def _candidate_mask(
    gray: np.ndarray,
    raw_mask: np.ndarray | None,
    detector_refined_mask: np.ndarray | None,
    config: CleanerConfig,
) -> tuple[np.ndarray, str]:
    raw = (raw_mask > 0).astype(np.uint8) * 255 if raw_mask is not None and np.count_nonzero(raw_mask) else None
    refined = (
        (detector_refined_mask > 0).astype(np.uint8) * 255
        if detector_refined_mask is not None and np.count_nonzero(detector_refined_mask)
        else None
    )
    threshold = (gray < 190).astype(np.uint8) * 255
    source = config.mask_source
    if source == "auto":
        if config.mask_refine_mode == "threshold":
            source = "threshold"
        elif config.provider.startswith("lama-large") or config.quality_preset == "quality" or config.mask_refine_mode == "mit_fit_text":
            source = "union" if raw is not None and refined is not None else ("refined" if refined is not None else ("raw" if raw is not None else "threshold"))
        else:
            source = "raw" if raw is not None else ("refined" if refined is not None else "threshold")

    if source == "raw":
        return (raw if raw is not None else np.zeros_like(gray, dtype=np.uint8)), "raw"
    if source == "refined":
        return (refined if refined is not None else (raw if raw is not None else threshold)), "refined" if refined is not None else ("raw" if raw is not None else "threshold")
    if source == "union":
        masks = [mask for mask in (raw, refined) if mask is not None]
        if masks:
            union = masks[0].copy()
            for mask in masks[1:]:
                union = np.maximum(union, mask)
            return union, "union"
        return threshold, "threshold"
    return threshold, "threshold"


def _load_mask(path: Path | None, width: int, height: int) -> np.ndarray | None:
    if path is None:
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    if raw.shape[:2] != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
    return raw


def _collect_text_polygons(blocks: list[CleanerBlock], width: int, height: int) -> list[TextPolygon]:
    polygons: list[TextPolygon] = []
    for block in blocks:
        raw_polygons = block.textline_polygons or ([block.polygon] if block.polygon else [])
        if not raw_polygons:
            boxes = block.textline_bboxes or [block.mask_bbox or block.bbox]
            raw_polygons = [_bbox_to_polygon(box) for box in boxes]
        for polygon in raw_polygons:
            if polygon:
                pixel_polygon = _polygon_to_pixels(polygon, width, height)
                _, _, w, h = cv2.boundingRect(pixel_polygon)
                polygons.append(TextPolygon(block.id, pixel_polygon, float(max(1, min(w, h)))))
    return polygons


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _polygon_to_pixels(polygon: list[list[float]], width: int, height: int) -> np.ndarray:
    points = []
    for x, y in polygon:
        points.append([int(round(max(0.0, min(1.0, x)) * width)), int(round(max(0.0, min(1.0, y)) * height))])
    return np.array(points, dtype=np.int32)


def _expanded_rect_for_polygon(polygon: np.ndarray, width: int, height: int, offset: int) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(polygon)
    return (
        max(0, x - offset),
        max(0, y - offset),
        min(width, x + w + offset),
        min(height, y + h + offset),
    )


def _polygon_mask_for_rect(polygon: np.ndarray, rect: tuple[int, int, int, int], offset: int) -> np.ndarray:
    left, top, right, bottom = rect
    mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    local_polygon = polygon.copy()
    local_polygon[:, 0] -= left
    local_polygon[:, 1] -= top
    cv2.fillPoly(mask, [local_polygon], 255)
    if offset > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, offset // 2 * 2 + 1), max(3, offset // 2 * 2 + 1)))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _clip_candidate_to_polygons(
    candidate: np.ndarray,
    polygons: list[TextPolygon],
    config: CleanerConfig,
    width: int,
    height: int,
) -> np.ndarray:
    safe = np.zeros_like(candidate)
    for text_polygon in polygons:
        polygon = text_polygon.polygon
        left, top, right, bottom = _expanded_rect_for_polygon(polygon, width, height, config.mask_dilation_offset)
        safe[top:bottom, left:right] = np.maximum(
            safe[top:bottom, left:right],
            _polygon_mask_for_rect(polygon, (left, top, right, bottom), config.mask_dilation_offset),
        )
    return cv2.bitwise_and(candidate, safe)


def _refine_mit_fit_text(
    candidate: np.ndarray,
    text_polygons: list[TextPolygon],
    config: CleanerConfig,
    width: int,
    height: int,
    mask_source: str,
    raw_candidate_pixels: int,
) -> RefinedMaskResult:
    work = (candidate > 0).astype(np.uint8)
    for text_polygon in text_polygons:
        x, y, w, h = cv2.boundingRect(text_polygon.polygon)
        cv2.rectangle(work, (max(0, x), max(0, y)), (min(width - 1, x + w), min(height - 1, y + h)), 0, 1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(work, 8)
    refined = np.zeros_like(candidate)
    component_count = max(0, num_labels - 1)
    assigned_count = 0
    rescued_unassigned_count = 0
    removed_count = 0
    reject_reasons: dict[str, int] = {}
    polygon_metrics: dict[str, dict[str, Any]] = {
        f"{text_polygon.block_id}:{index}": {
            "block_id": text_polygon.block_id,
            "polygon_index": index + 1,
            "assigned_component_count": 0,
            "assigned_pixels": 0,
            "font_size": text_polygon.font_size,
        }
        for index, text_polygon in enumerate(text_polygons)
    }
    polygon_masks: list[tuple[str, TextPolygon, np.ndarray, int]] = []
    for index, text_polygon in enumerate(text_polygons):
        poly_mask = np.zeros_like(candidate)
        cv2.fillPoly(poly_mask, [text_polygon.polygon], 255)
        if config.mask_dilation_offset > 0:
            dynamic = _dynamic_text_kernel(text_polygon.font_size, config.mask_dilation_offset)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dynamic, dynamic))
            poly_mask = cv2.dilate(poly_mask, kernel, iterations=1)
        polygon_masks.append((f"{text_polygon.block_id}:{index}", text_polygon, poly_mask, int(np.count_nonzero(poly_mask))))

    image_area = max(1, width * height)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.component_min_area:
            removed_count += 1
            _count(reject_reasons, "too_small")
            continue
        component_bbox_area = max(1, int(stats[label, cv2.CC_STAT_WIDTH]) * int(stats[label, cv2.CC_STAT_HEIGHT]))
        component = (labels == label).astype(np.uint8) * 255
        best: tuple[str, TextPolygon, float, int] | None = None
        for key, text_polygon, poly_mask, poly_area in polygon_masks:
            overlap = int(np.count_nonzero(cv2.bitwise_and(component, poly_mask)))
            if overlap <= 0:
                continue
            score = overlap / max(1, min(area, poly_area))
            if best is None or score > best[2]:
                best = (key, text_polygon, score, overlap)

        cx, cy = centroids[label]
        if best is None:
            best_distance: tuple[str, TextPolygon, float] | None = None
            for key, text_polygon, _poly_mask, _poly_area in polygon_masks:
                x, y, w, h = cv2.boundingRect(text_polygon.polygon)
                px = min(max(float(cx), float(x)), float(x + w))
                py = min(max(float(cy), float(y)), float(y + h))
                distance = float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
                tolerance = max(text_polygon.font_size * 1.2, float(config.mask_dilation_offset), 8.0)
                if distance <= tolerance and (best_distance is None or distance < best_distance[2]):
                    best_distance = (key, text_polygon, distance)
            if best_distance is not None:
                best = (best_distance[0], best_distance[1], 0.0, 0)

        huge_component = area / image_area > 0.18 or area / component_bbox_area > 0.94
        if best is None:
            if _can_rescue_unassigned_component(mask_source, area, image_area, component_bbox_area):
                dynamic = max(3, min(21, config.mask_dilation_offset // 2 * 2 + 1))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dynamic, dynamic))
                refined = np.maximum(refined, cv2.dilate(component, kernel, iterations=1))
                rescued_unassigned_count += 1
                continue
            removed_count += 1
            _count(reject_reasons, "unassigned_or_background")
            continue
        if huge_component and best[2] < config.component_min_overlap_ratio:
            removed_count += 1
            _count(reject_reasons, "unassigned_or_background")
            continue

        _, text_polygon, score, overlap = best
        dynamic = _dynamic_text_kernel(text_polygon.font_size, config.mask_dilation_offset)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dynamic, dynamic))
        expanded = cv2.dilate(component, kernel, iterations=1)
        refined = np.maximum(refined, expanded)
        assigned_count += 1
        metric = polygon_metrics[best[0]]
        metric["assigned_component_count"] += 1
        metric["assigned_pixels"] += area
        metric["best_overlap_ratio"] = max(float(metric.get("best_overlap_ratio", 0.0)), float(score))
        metric["best_overlap_pixels"] = max(int(metric.get("best_overlap_pixels", 0)), overlap)

    refined = cv2.bitwise_and(refined, candidate)
    refined = _post_dilate(refined, config)
    refined_pixels = int(np.count_nonzero(refined))
    missed = cv2.bitwise_and(candidate, cv2.bitwise_not((refined > 0).astype(np.uint8) * 255))
    residual_candidate_pixels = int(np.count_nonzero(missed))
    stats = _stats(
        config.mask_refine_mode,
        mask_source,
        raw_candidate_pixels,
        refined_pixels,
        residual_candidate_pixels,
        component_count,
        removed_count,
        reject_reasons,
        [],
    )
    stats.update(
        {
            "text_polygon_count": len(text_polygons),
            "assigned_component_count": assigned_count,
            "rescued_unassigned_component_count": rescued_unassigned_count,
            "unassigned_component_count": max(0, component_count - assigned_count - rescued_unassigned_count - removed_count),
            "block_metrics": list(polygon_metrics.values()),
            "residual_candidate_ratio": residual_candidate_pixels / raw_candidate_pixels if raw_candidate_pixels else 0.0,
            "final_to_candidate_ratio": refined_pixels / raw_candidate_pixels if raw_candidate_pixels else 0.0,
        }
    )
    if stats["residual_candidate_ratio"] > 0.18:
        stats["warning_codes"].append("UNMASKED_TEXT_CANDIDATE_HIGH")
    return RefinedMaskResult(mask=refined, stats=_with_image_stats(stats, width, height, refined_pixels))


def _dynamic_text_kernel(font_size: float, offset: int) -> int:
    size = int(round(max(3.0, min(41.0, font_size * 0.28 + offset * 0.45))))
    if size % 2 == 0:
        size += 1
    return size


def _can_rescue_unassigned_component(mask_source: str, area: int, image_area: int, component_bbox_area: int) -> bool:
    if mask_source not in {"raw", "refined", "union"}:
        return False
    if area / max(1, image_area) > 0.025:
        return False
    fill_ratio = area / max(1, component_bbox_area)
    if fill_ratio > 0.96:
        return False
    return True


def _filter_components(candidate: np.ndarray, safe_mask: np.ndarray, config: CleanerConfig) -> tuple[np.ndarray, dict[str, Any]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), 8)
    refined = np.zeros_like(candidate)
    removed = 0
    reasons: dict[str, int] = {}
    crop_area = max(1, candidate.shape[0] * candidate.shape[1])
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.component_min_area:
            removed += 1
            _count(reasons, "too_small")
            continue
        if area / crop_area > config.component_max_area_ratio:
            removed += 1
            _count(reasons, "too_large")
            continue
        component = (labels == label).astype(np.uint8) * 255
        overlap = int(np.count_nonzero(cv2.bitwise_and(component, safe_mask)))
        overlap_ratio = overlap / max(1, area)
        cx, cy = centroids[label]
        center_inside = 0 <= int(cy) < safe_mask.shape[0] and 0 <= int(cx) < safe_mask.shape[1] and safe_mask[int(cy), int(cx)] > 0
        if overlap_ratio < config.component_min_overlap_ratio and not center_inside:
            removed += 1
            _count(reasons, "outside_textline_polygon")
            continue
        refined = np.maximum(refined, component)
    return refined, {
        "component_count": max(0, num_labels - 1),
        "removed_component_count": removed,
        "component_reject_reasons": reasons,
        "refined_pixels": int(np.count_nonzero(refined)),
    }


def _post_dilate(mask: np.ndarray, config: CleanerConfig) -> np.ndarray:
    if not np.count_nonzero(mask):
        return mask
    size = config.kernel_size
    if config.mask_dilation_offset > 0:
        size = max(size, min(17, config.mask_dilation_offset // 4 * 2 + 1))
    if size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel, iterations=1)


def _stats(
    mode: str,
    mask_source: str,
    raw_candidate_pixels: int,
    refined_pixels: int,
    residual_candidate_pixels: int,
    component_count: int,
    removed_component_count: int,
    reject_reasons: dict[str, int],
    warning_codes: list[str],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "mask_source": mask_source,
        "raw_candidate_pixels": raw_candidate_pixels,
        "refined_mask_pixels": refined_pixels,
        "residual_candidate_pixels": residual_candidate_pixels,
        "component_count": component_count,
        "removed_component_count": removed_component_count,
        "component_reject_reasons": reject_reasons,
        "residual_candidate_ratio": residual_candidate_pixels / raw_candidate_pixels if raw_candidate_pixels else 0.0,
        "warning_codes": warning_codes,
        "block_metrics": [],
    }


def _with_image_stats(stats: dict[str, Any], width: int, height: int, refined_pixels: int) -> dict[str, Any]:
    image_pixels = width * height
    stats.update(
        {
            "image_width": width,
            "image_height": height,
            "image_pixels": image_pixels,
            "mask_coverage_ratio": refined_pixels / image_pixels if image_pixels else 0.0,
        }
    )
    return stats


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _count(target: dict[str, int], key: str) -> None:
    target[key] = target.get(key, 0) + 1
