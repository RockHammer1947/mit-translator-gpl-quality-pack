from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from comic_detector_sidecar.model_manager import default_comic_text_detector_model_path


def detect_mit_ctd(
    *,
    image_path: Path,
    model_path: Path | None,
    detection_size: int,
    text_threshold: float,
    box_threshold: float,
    unclip_ratio: float,
    nms_threshold: float,
    min_textline_area_ratio: float,
    max_regions: int,
) -> dict[str, Any]:
    # GPL-derived from MIT comic-text-detector ONNX inference path.
    resolved_model = model_path or default_comic_text_detector_model_path()
    if not image_path.exists():
        raise FileNotFoundError(f"input image not found: {image_path}")
    if not resolved_model.exists():
        raise FileNotFoundError(f"comic-text-detector ONNX model not found: {resolved_model}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to load image: {image_path}")
    height, width = image.shape[:2]

    # The distributed ONNX graph has a fixed 1024 input. Higher coverage presets
    # are implemented by lower thresholds and post-processing rather than resizing
    # the graph input to an unsupported shape.
    input_size = 1024
    image_in, ratio, dw, dh = _letterbox(image, input_size)
    model = _TextDetBaseDNN(input_size, resolved_model)
    block_pred, mask, lines_map = model(image_in)
    if mask.shape[1] == 2:
        mask, lines_map = lines_map, mask

    raw_mask = _postprocess_mask(mask)
    raw_mask = raw_mask[: raw_mask.shape[0] - dh, : raw_mask.shape[1] - dw]
    raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    line_pred = lines_map[..., : lines_map.shape[2] - dh, : lines_map.shape[3] - dw]
    representer = _SegDetectorRepresenter(
        thresh=max(0.05, min(0.95, box_threshold)),
        box_thresh=max(0.05, min(0.95, text_threshold)),
        max_candidates=1000,
        unclip_ratio=max(0.5, min(4.0, unclip_ratio)),
    )
    line_polys, line_scores = representer(line_pred, height=height, width=width)
    textlines = _regions_from_polygons(
        line_polys[0],
        line_scores[0],
        width=width,
        height=height,
        min_area_ratio=min_textline_area_ratio,
        prefix="line",
        kind="text_line",
        source="mit_ctd_db_map",
    )
    textlines = _dedupe_regions(textlines, iou_threshold=0.58, max_regions=max(max_regions * 4, max_regions))

    regions = _decode_block_regions(
        block_pred[0],
        width=width,
        height=height,
        ratio=ratio,
        dw=dw,
        dh=dh,
        confidence_threshold=max(0.05, min(0.95, text_threshold)),
        nms_threshold=nms_threshold,
        max_regions=max_regions,
    )
    if not regions:
        regions = _regions_from_textline_clusters(textlines, max_regions=max_regions)

    return {
        "schema_version": "mit_ctd_adapter.v1",
        "provider": "mit-ctd",
        "model_path": str(resolved_model),
        "effective_detection_size": input_size,
        "requested_detection_size": detection_size,
        "regions": regions,
        "textlines": textlines,
        "raw_mask": raw_mask,
    }


class _TextDetBaseDNN:
    def __init__(self, input_size: int, model_path: Path):
        self.input_size = input_size
        self.model = cv2.dnn.readNetFromONNX(str(model_path))
        self.output_names = self.model.getUnconnectedOutLayersNames()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(self.input_size, self.input_size))
        self.model.setInput(blob)
        block_pred, mask, lines_map = self.model.forward(self.output_names)
        return block_pred, mask, lines_map


class _SegDetectorRepresenter:
    def __init__(self, thresh: float, box_thresh: float, max_candidates: int, unclip_ratio: float):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, pred: np.ndarray, *, height: int, width: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        boxes_batch: list[np.ndarray] = []
        scores_batch: list[np.ndarray] = []
        for batch_index in range(pred.shape[0]):
            boxes, scores = self._boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def _boxes_from_bitmap(self, pred: np.ndarray, bitmap: np.ndarray, dest_width: int, dest_height: int) -> tuple[np.ndarray, np.ndarray]:
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[np.ndarray] = []
        scores: list[float] = []
        for contour in contours[: self.max_candidates]:
            contour = contour.squeeze(1)
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue
            points, short_side = self._get_mini_boxes(contour)
            if short_side < self.min_size:
                continue
            score = self._box_score_fast(pred, contour)
            if score < self.box_thresh:
                continue
            expanded = self._unclip(np.array(points), self.unclip_ratio)
            if expanded.size == 0:
                continue
            box, short_side = self._get_mini_boxes(expanded.reshape((-1, 1, 2)))
            if short_side < self.min_size:
                continue
            box_arr = np.array(box)
            box_arr[:, 0] = np.clip(np.round(box_arr[:, 0] / pred.shape[1] * dest_width), 0, dest_width)
            box_arr[:, 1] = np.clip(np.round(box_arr[:, 1] / pred.shape[0] * dest_height), 0, dest_height)
            boxes.append(box_arr.astype(np.int64))
            scores.append(float(score))
        if not boxes:
            return np.zeros((0, 4, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        return np.stack(boxes), np.array(scores, dtype=np.float32)

    def _unclip(self, box: np.ndarray, unclip_ratio: float) -> np.ndarray:
        rect = cv2.minAreaRect(box.astype(np.float32))
        points = cv2.boxPoints(rect)
        cx, cy = np.mean(points, axis=0)
        return (points - [cx, cy]) * unclip_ratio + [cx, cy]

    @staticmethod
    def _get_mini_boxes(contour: np.ndarray) -> tuple[list[np.ndarray], float]:
        bounding_box = cv2.minAreaRect(contour.astype(np.float32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda item: item[0])
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2
        return [points[index_1], points[index_2], points[index_3], points[index_4]], float(min(bounding_box[1]))

    @staticmethod
    def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        shifted = box.copy()
        shifted[:, 0] -= xmin
        shifted[:, 1] -= ymin
        cv2.fillPoly(mask, shifted.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def _letterbox(image: np.ndarray, input_size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    ratio = min(input_size / max(1, height), input_size / max(1, width))
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    dw = input_size - new_width
    dh = input_size - new_height
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[:new_height, :new_width] = resized
    return canvas, ratio, dw, dh


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.squeeze(mask)
    mask = np.clip(mask * 255, 0, 255)
    return mask.astype(np.uint8)


def _regions_from_polygons(
    polygons: np.ndarray,
    scores: np.ndarray,
    *,
    width: int,
    height: int,
    min_area_ratio: float,
    prefix: str,
    kind: str,
    source: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    min_area = max(8.0, width * height * min_area_ratio)
    for index, (poly, score) in enumerate(zip(polygons, scores), 1):
        x1 = float(np.min(poly[:, 0]))
        y1 = float(np.min(poly[:, 1]))
        x2 = float(np.max(poly[:, 0]))
        y2 = float(np.max(poly[:, 1]))
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        bbox = _normalize_xyxy([x1, y1, x2, y2], width, height)
        if bbox is None:
            continue
        items.append(
            {
                "id": f"{prefix}_{len(items) + 1:03d}",
                "bbox": bbox,
                "confidence": round(float(score), 4),
                "kind": kind,
                "reading_order": len(items) + 1,
                "polygon": [[round(float(x) / width, 6), round(float(y) / height, 6)] for x, y in poly[:4]],
                "metadata": {"source": source},
            }
        )
    return sorted(items, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _decode_block_regions(
    pred: np.ndarray,
    *,
    width: int,
    height: int,
    ratio: float,
    dw: int,
    dh: int,
    confidence_threshold: float,
    nms_threshold: float,
    max_regions: int,
) -> list[dict[str, Any]]:
    if pred.size == 0:
        return []
    object_conf = pred[:, 4]
    class_conf = pred[:, 5:].max(axis=1) if pred.shape[1] > 5 else np.ones_like(object_conf)
    scores = object_conf * class_conf
    keep = scores >= confidence_threshold
    pred = pred[keep]
    scores = scores[keep]
    if pred.size == 0:
        return []
    boxes = np.column_stack([pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2, pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2])
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, 1024 - dw)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, 1024 - dh)
    boxes = boxes / max(ratio, 1e-6)
    selected = _nms(boxes, scores, nms_threshold)[:max_regions]
    regions: list[dict[str, Any]] = []
    for index in selected:
        bbox = _normalize_xyxy(boxes[index], width, height)
        if bbox is None:
            continue
        regions.append(
            {
                "id": f"reg_{len(regions) + 1:03d}",
                "bbox": bbox,
                "confidence": round(float(scores[index]), 4),
                "kind": "text_region",
                "reading_order": len(regions) + 1,
                "polygon": _bbox_to_polygon(bbox),
                "metadata": {"source": "mit_ctd_block_head"},
            }
        )
    return sorted(regions, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _regions_from_textline_clusters(textlines: list[dict[str, Any]], *, max_regions: int) -> list[dict[str, Any]]:
    regions = []
    for index, line in enumerate(textlines[:max_regions], 1):
        regions.append(
            {
                "id": f"reg_{index:03d}",
                "bbox": line["bbox"],
                "confidence": line["confidence"],
                "kind": "text_region",
                "reading_order": index,
                "polygon": line["polygon"],
                "metadata": {"source": "mit_ctd_textline_promoted_region"},
            }
        )
    return regions


def _dedupe_regions(items: list[dict[str, Any]], *, iou_threshold: float, max_regions: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: (-value["confidence"], value["bbox"][1], value["bbox"][0])):
        if any(_bbox_iou(item["bbox"], existing["bbox"]) > iou_threshold for existing in kept):
            continue
        kept.append(item)
        if len(kept) >= max_regions:
            break
    kept.sort(key=lambda value: (value["bbox"][1], value["bbox"][0]))
    for index, item in enumerate(kept, 1):
        item["id"] = f"line_{index:03d}" if item["kind"] == "text_line" else f"reg_{index:03d}"
        item["reading_order"] = index
    return kept


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = np.array([_xyxy_iou(boxes[current], boxes[int(index)]) for index in order[1:]])
        order = order[1:][ious <= threshold]
    return keep


def _normalize_xyxy(box: np.ndarray | list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(width, x1))
    x2 = max(0.0, min(width, x2))
    y1 = max(0.0, min(height, y1))
    y2 = max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1 / width, 6), round(y1 / height, 6), round((x2 - x1) / width, 6), round((y2 - y1) / height, 6)]


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _bbox_iou(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return _xyxy_iou(np.array([lx, ly, lx + lw, ly + lh]), np.array([rx, ry, rx + rw, ry + rh]))


def _xyxy_iou(left: np.ndarray, right: np.ndarray) -> float:
    ix1 = max(float(left[0]), float(right[0]))
    iy1 = max(float(left[1]), float(right[1]))
    ix2 = min(float(left[2]), float(right[2]))
    iy2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - inter
    return inter / union if union > 0 else 0.0
