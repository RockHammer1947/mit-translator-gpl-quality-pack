from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from comic_detector_sidecar.model_manager import default_comic_text_detector_model_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean-room comic-text-detector ONNX adapter")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-mask-output", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--detection-size", type=int, default=1024)
    parser.add_argument("--text-threshold", type=float, default=0.5)
    parser.add_argument("--box-threshold", type=float, default=0.45)
    parser.add_argument("--unclip-ratio", type=float, default=2.3)
    args = parser.parse_args()

    try:
        payload = detect(
            image_path=Path(args.input),
            model_path=Path(args.model_path).expanduser() if args.model_path else default_comic_text_detector_model_path(),
            raw_mask_output=Path(args.raw_mask_output),
            detection_size=args.detection_size,
            text_threshold=args.text_threshold,
            box_threshold=args.box_threshold,
            unclip_ratio=args.unclip_ratio,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


def detect(
    *,
    image_path: Path,
    model_path: Path,
    raw_mask_output: Path,
    detection_size: int,
    text_threshold: float,
    box_threshold: float,
    unclip_ratio: float,
) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"input image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"comic-text-detector ONNX model not found: {model_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to load image: {image_path}")
    height, width = image.shape[:2]
    net = cv2.dnn.readNetFromONNX(str(model_path))
    size = max(512, min(2048, int(detection_size)))
    canvas, scale, new_width, new_height = _letterbox_top_left(image, size)
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, size=(size, size))
    net.setInput(blob)
    block_pred, line_map, text_mask = net.forward(net.getUnconnectedOutLayersNames())

    raw_mask = _restore_probability_map(text_mask[0, 0], width, height, new_width, new_height)
    mask_u8 = (raw_mask >= max(0.1, min(0.9, text_threshold * 0.6))).astype(np.uint8) * 255
    raw_mask_output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(raw_mask_output), mask_u8)

    regions = _decode_block_predictions(
        block_pred[0],
        width=width,
        height=height,
        scale=scale,
        confidence_threshold=max(0.05, min(0.95, text_threshold)),
        nms_threshold=0.35,
        max_regions=96,
    )
    textlines = _decode_textlines_from_maps(
        line_map=line_map[0, 0],
        raw_mask=raw_mask,
        width=width,
        height=height,
        new_width=new_width,
        new_height=new_height,
        box_threshold=max(0.05, min(0.95, box_threshold)),
        unclip_ratio=unclip_ratio,
    )
    if not regions and textlines:
        regions = [_region_from_textline(line, index) for index, line in enumerate(textlines, 1)]
    return {
        "schema_version": "comic_text_detector_adapter.v1",
        "provider": "comic-text-detector",
        "model_path": str(model_path),
        "regions": regions,
        "textlines": textlines,
    }


def _letterbox_top_left(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:new_height, :new_width] = resized
    return canvas, scale, new_width, new_height


def _restore_probability_map(map_data: np.ndarray, width: int, height: int, new_width: int, new_height: int) -> np.ndarray:
    cropped = map_data[:new_height, :new_width]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def _decode_block_predictions(
    pred: np.ndarray,
    *,
    width: int,
    height: int,
    scale: float,
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
    boxes = np.column_stack(
        [
            pred[:, 0] - pred[:, 2] / 2,
            pred[:, 1] - pred[:, 3] / 2,
            pred[:, 0] + pred[:, 2] / 2,
            pred[:, 1] + pred[:, 3] / 2,
        ]
    )
    boxes = boxes / max(scale, 1e-6)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    selected = _nms(boxes, scores, nms_threshold)[:max_regions]
    regions = []
    for reading_order, index in enumerate(selected, 1):
        bbox = _normalize_xyxy(boxes[index], width, height)
        if bbox is None:
            continue
        regions.append(
            {
                "id": f"reg_{reading_order:03d}",
                "bbox": bbox,
                "confidence": float(round(float(scores[index]), 4)),
                "kind": "text_region",
                "reading_order": reading_order,
                "polygon": _bbox_to_polygon(bbox),
                "metadata": {"source": "comic_text_detector_onnx_block"},
            }
        )
    return regions


def _decode_textlines_from_maps(
    *,
    line_map: np.ndarray,
    raw_mask: np.ndarray,
    width: int,
    height: int,
    new_width: int,
    new_height: int,
    box_threshold: float,
    unclip_ratio: float,
) -> list[dict[str, Any]]:
    restored = _restore_probability_map(line_map, width, height, new_width, new_height)
    binary = (restored >= box_threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    image_area = max(1, width * height)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(24, image_area * 0.00003):
            continue
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect)
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        if w < 3 or h < 3 or w * h > image_area * 0.20:
            continue
        padded = _pad_xyxy([x, y, x + w, y + h], width, height, unclip_ratio)
        bbox = _normalize_xyxy(np.array(padded, dtype=np.float32), width, height)
        if bbox is None:
            continue
        mask_slice = raw_mask[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)]
        score = float(np.mean(mask_slice)) if mask_slice.size else float(np.mean(restored[binary > 0]))
        candidates.append((bbox, max(score, box_threshold)))
    candidates.sort(key=lambda item: (item[0][1], item[0][0]))
    deduped: list[tuple[list[float], float]] = []
    for bbox, score in candidates:
        if any(_bbox_iou(bbox, existing[0]) > 0.55 for existing in deduped):
            continue
        deduped.append((bbox, score))
    return [
        {
            "id": f"line_{index:03d}",
            "bbox": bbox,
            "confidence": float(round(score, 4)),
            "kind": "text_line",
            "reading_order": index,
            "polygon": _bbox_to_polygon(bbox),
            "metadata": {"source": "comic_text_detector_onnx_line_map"},
        }
        for index, (bbox, score) in enumerate(deduped, 1)
    ]


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


def _bbox_iou(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return _xyxy_iou(
        np.array([lx, ly, lx + lw, ly + lh], dtype=np.float32),
        np.array([rx, ry, rx + rw, ry + rh], dtype=np.float32),
    )


def _normalize_xyxy(box: np.ndarray | list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [
        round(x1 / width, 6),
        round(y1 / height, 6),
        round((x2 - x1) / width, 6),
        round((y2 - y1) / height, 6),
    ]


def _pad_xyxy(box: list[int], width: int, height: int, unclip_ratio: float) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    pad = max(1.0, min(width, height) * 0.0025 * max(1.0, unclip_ratio))
    return [max(0.0, x1 - pad), max(0.0, y1 - pad), min(float(width), x2 + pad), min(float(height), y2 + pad)]


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _region_from_textline(line: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"reg_{index:03d}",
        "bbox": line["bbox"],
        "confidence": line["confidence"],
        "kind": "text_region",
        "reading_order": index,
        "polygon": line["polygon"],
        "metadata": {"source": "textline_promoted_region"},
    }


if __name__ == "__main__":
    raise SystemExit(main())
