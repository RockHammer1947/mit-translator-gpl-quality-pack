from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_image_cv2(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def crop_bbox(image, bbox: list[float]):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = pixel_rect_from_bbox(bbox, width, height)
    return image[y1:y2, x1:x2]


def pixel_rect_from_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x1 = max(0, min(width, int(round(x * width))))
    y1 = max(0, min(height, int(round(y * height))))
    x2 = max(0, min(width, int(round((x + w) * width))))
    y2 = max(0, min(height, int(round((y + h) * height))))
    return x1, y1, x2, y2


def bbox_from_polygon(polygon: list[list[float]]) -> list[float]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    left = max(0.0, min(xs))
    top = max(0.0, min(ys))
    right = min(1.0, max(xs))
    bottom = min(1.0, max(ys))
    return [left, top, max(0.0, right - left), max(0.0, bottom - top)]


def crop_for_ocr(
    image,
    *,
    region: list[float] | None = None,
    polygon: list[list[float]] | None = None,
    crop_policy: str | None = None,
    orientation: str | None = None,
    target_text_height: int | None = None,
    padding_ratio: float | None = None,
    preprocess: str | None = None,
) -> tuple[Any, list[float], list[list[float]] | None, dict[str, Any]]:
    height, width = image.shape[:2]
    resolved_policy = resolve_crop_policy(crop_policy, region=region, polygon=polygon)
    resolved_padding = float(padding_ratio or 0.0)
    transform_metadata: dict[str, Any] = {}
    if resolved_policy == "mit_textline" and polygon:
        crop, transform_metadata = crop_mit_textline(
            image,
            polygon,
            orientation=orientation,
            target_text_height=target_text_height or 48,
        )
        crop = add_crop_padding(crop, resolved_padding)
        source_bbox = bbox_from_polygon(polygon)
        source_polygon = polygon
    elif resolved_policy == "polygon_perspective" and polygon:
        crop = crop_polygon_perspective(image, polygon)
        crop = add_crop_padding(crop, resolved_padding)
        source_bbox = bbox_from_polygon(polygon)
        source_polygon = polygon
    else:
        source_bbox = region or (bbox_from_polygon(polygon) if polygon else [0.0, 0.0, 1.0, 1.0])
        source_bbox = expand_normalized_bbox(source_bbox, resolved_padding)
        crop = crop_bbox(image, source_bbox)
        source_polygon = polygon or bbox_to_polygon(source_bbox)
        resolved_policy = "bbox"

    original_shape = crop.shape[:2] if crop is not None and crop.size else (0, 0)
    if resolved_policy != "mit_textline":
        crop = resize_to_text_height(crop, target_text_height)
    crop = preprocess_crop(crop, preprocess)
    resized_shape = crop.shape[:2] if crop is not None and crop.size else (0, 0)
    metadata = {
        "crop_policy": resolved_policy,
        "requested_crop_policy": crop_policy,
        "target_text_height": target_text_height,
        "padding_ratio": resolved_padding,
        "preprocess": preprocess or "none",
        "orientation": orientation,
        "source_bbox": source_bbox,
        "source_polygon": source_polygon,
        "image_width": width,
        "image_height": height,
        "crop_width": int(resized_shape[1]),
        "crop_height": int(resized_shape[0]),
        "original_crop_width": int(original_shape[1]),
        "original_crop_height": int(original_shape[0]),
        **transform_metadata,
    }
    return crop, source_bbox, source_polygon, metadata


def resolve_crop_policy(
    crop_policy: str | None,
    *,
    region: list[float] | None,
    polygon: list[list[float]] | None,
) -> str:
    if crop_policy and crop_policy != "adaptive":
        return crop_policy
    if not polygon:
        return "bbox"
    bbox = bbox_from_polygon(polygon)
    _, _, bw, bh = bbox
    if bw <= 0.0 or bh <= 0.0:
        return "bbox"
    # Perspective crops help skewed or narrow textlines, but tiny or very
    # elongated polygons often lose context. MIT-style manga OCR benefits from
    # falling back to padded bbox crops for those cases.
    aspect = max(bw / bh, bh / bw)
    area = bw * bh
    if area < 0.0008 or aspect > 8.0:
        return "bbox"
    return "polygon_perspective"


def crop_mit_textline(
    image,
    polygon: list[list[float]],
    *,
    orientation: str | None = None,
    target_text_height: int = 48,
) -> tuple[Any, dict[str, Any]]:
    """MIT-style textline crop.

    MIT's 48px recognizer normalizes each detected textline into a 48px-high
    horizontal strip. For vertical Japanese textlines, the source column is
    first perspective-warped upright and then rotated so top-to-bottom reading
    order becomes left-to-right recognizer input.
    """
    height, width = image.shape[:2]
    points = normalized_polygon_to_pixels(polygon, width, height)
    if points.shape[0] > 4:
        rect = cv2.minAreaRect(points.astype(np.float32))
        points = cv2.boxPoints(rect)
    ordered = order_quad_points(points.astype(np.float32))
    top_w = np.linalg.norm(ordered[1] - ordered[0])
    bottom_w = np.linalg.norm(ordered[2] - ordered[3])
    left_h = np.linalg.norm(ordered[3] - ordered[0])
    right_h = np.linalg.norm(ordered[2] - ordered[1])
    src_w = max(1.0, float(max(top_w, bottom_w)))
    src_h = max(1.0, float(max(left_h, right_h)))
    resolved_orientation = infer_textline_orientation(
        orientation=orientation,
        width=src_w,
        height=src_h,
    )
    text_height = max(8, int(target_text_height or 48))
    if resolved_orientation == "vertical":
        warp_w = text_height
        warp_h = max(1, int(round(src_h / src_w * warp_w)))
    else:
        warp_h = text_height
        warp_w = max(1, int(round(src_w / src_h * warp_h)))
    destination = np.array(
        [[0, 0], [warp_w - 1, 0], [warp_w - 1, warp_h - 1], [0, warp_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (warp_w, warp_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if resolved_orientation == "vertical":
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    metadata = {
        "mit_textline_orientation": resolved_orientation,
        "mit_textline_source_width": src_w,
        "mit_textline_source_height": src_h,
        "mit_textline_warp_width": int(warp_w),
        "mit_textline_warp_height": int(warp_h),
        "mit_textline_rotated": resolved_orientation == "vertical",
    }
    return warped, metadata


def infer_textline_orientation(*, orientation: str | None, width: float, height: float) -> str:
    normalized = (orientation or "").strip().lower()
    if normalized.startswith("v"):
        return "vertical"
    if normalized.startswith("h"):
        return "horizontal"
    if normalized in {"unknown", "auto", ""}:
        return "vertical" if height > width * 1.15 else "horizontal"
    return "vertical" if height > width * 1.15 else "horizontal"


def expand_normalized_bbox(bbox: list[float], padding_ratio: float) -> list[float]:
    if padding_ratio <= 0:
        return bbox
    x, y, w, h = bbox
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio
    left = max(0.0, x - pad_x)
    top = max(0.0, y - pad_y)
    right = min(1.0, x + w + pad_x)
    bottom = min(1.0, y + h + pad_y)
    return [left, top, max(0.0, right - left), max(0.0, bottom - top)]


def add_crop_padding(crop, padding_ratio: float):
    if crop is None or crop.size == 0 or padding_ratio <= 0:
        return crop
    height, width = crop.shape[:2]
    pad_x = max(1, int(round(width * padding_ratio)))
    pad_y = max(1, int(round(height * padding_ratio)))
    return cv2.copyMakeBorder(crop, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def preprocess_crop(crop, preprocess: str | None):
    if crop is None or crop.size == 0 or preprocess in (None, "none"):
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    if preprocess in ("auto", "manga_enhance"):
        stddev = float(np.std(gray))
        mean = float(np.mean(gray))
        if preprocess == "manga_enhance" or stddev < 48.0 or mean < 120.0:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            sharpened = cv2.addWeighted(enhanced, 1.35, cv2.GaussianBlur(enhanced, (0, 0), 1.0), -0.35, 0)
            return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return crop


def crop_polygon_perspective(image, polygon: list[list[float]]):
    height, width = image.shape[:2]
    points = normalized_polygon_to_pixels(polygon, width, height)
    if points.shape[0] > 4:
        rect = cv2.minAreaRect(points.astype(np.float32))
        points = cv2.boxPoints(rect)
    ordered = order_quad_points(points.astype(np.float32))
    top_w = np.linalg.norm(ordered[1] - ordered[0])
    bottom_w = np.linalg.norm(ordered[2] - ordered[3])
    left_h = np.linalg.norm(ordered[3] - ordered[0])
    right_h = np.linalg.norm(ordered[2] - ordered[1])
    crop_w = max(1, int(round(max(top_w, bottom_w))))
    crop_h = max(1, int(round(max(left_h, right_h))))
    destination = np.array(
        [[0, 0], [crop_w - 1, 0], [crop_w - 1, crop_h - 1], [0, crop_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (crop_w, crop_h), flags=cv2.INTER_CUBIC)


def normalized_polygon_to_pixels(polygon: list[list[float]], width: int, height: int) -> np.ndarray:
    points = []
    for x, y in polygon:
        px = max(0, min(width - 1, int(round(float(x) * width))))
        py = max(0, min(height - 1, int(round(float(y) * height))))
        points.append([px, py])
    return np.array(points, dtype=np.float32)


def order_quad_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] != 4:
        raise ValueError("order_quad_points expects exactly four points")
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def resize_to_text_height(crop, target_text_height: int | None):
    if crop is None or crop.size == 0 or not target_text_height:
        return crop
    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        return crop
    scale = target_text_height / height
    target_width = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA
    return cv2.resize(crop, (target_width, int(target_text_height)), interpolation=interpolation)


def bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
