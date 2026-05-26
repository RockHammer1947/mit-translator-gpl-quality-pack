from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from comic_detector_sidecar.model_manager import default_ogkalu_rtdetr_model_path

LABELS = {
    0: "bubble",
    1: "text_bubble",
    2: "text_free",
}


def detect_ogkalu_rtdetr(
    *,
    image_path: Path,
    model_path: Path | None,
    score_threshold: float,
    nms_threshold: float,
    max_regions: int,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except Exception as error:
        raise RuntimeError("onnxruntime is required for ogkalu-rtdetr provider") from error

    resolved_model = model_path or default_ogkalu_rtdetr_model_path()
    if not image_path.exists():
        raise FileNotFoundError(f"input image not found: {image_path}")
    if not resolved_model.exists():
        raise FileNotFoundError(f"ogkalu RT-DETR ONNX model not found: {resolved_model}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to load image: {image_path}")
    height, width = image.shape[:2]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    tensor = np.transpose(resized, (2, 0, 1))[None, ...]
    target_sizes = np.array([[height, width]], dtype=np.int64)

    session = ort.InferenceSession(str(resolved_model), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"images": tensor, "orig_target_sizes": target_sizes})
    labels, boxes, scores = outputs
    labels = labels[0]
    boxes = boxes[0]
    scores = scores[0]

    candidates = []
    for label, box, score in zip(labels, boxes, scores):
        score_value = float(score)
        if score_value < score_threshold:
            continue
        label_id = int(label)
        normalized = _normalize_xyxy(box, width=width, height=height)
        if normalized is None:
            continue
        candidates.append(
            {
                "label": label_id,
                "label_name": LABELS.get(label_id, f"class_{label_id}"),
                "bbox": normalized,
                "confidence": round(score_value, 4),
            }
        )

    text_candidates = [item for item in candidates if item["label"] in {1, 2}]
    bubble_candidates = [item for item in candidates if item["label"] == 0]
    text_candidates = _dedupe(text_candidates, nms_threshold=nms_threshold, max_regions=max(max_regions * 4, max_regions))
    bubble_candidates = _dedupe(bubble_candidates, nms_threshold=nms_threshold, max_regions=max_regions)

    textlines = []
    for index, item in enumerate(_sort_reading_order(text_candidates), 1):
        textlines.append(
            {
                "id": f"line_{index:03d}",
                "bbox": item["bbox"],
                "confidence": item["confidence"],
                "kind": "text_line",
                "reading_order": index,
                "polygon": _bbox_to_polygon(item["bbox"]),
                "metadata": {
                    "source": "ogkalu_rtdetr",
                    "label": item["label_name"],
                    "license": "Apache-2.0 model card; source data requires due diligence",
                },
            }
        )

    regions = []
    for index, item in enumerate(_sort_reading_order(text_candidates[:max_regions]), 1):
        regions.append(
            {
                "id": f"reg_{index:03d}",
                "bbox": item["bbox"],
                "confidence": item["confidence"],
                "kind": "text_region",
                "reading_order": index,
                "polygon": _bbox_to_polygon(item["bbox"]),
                "metadata": {
                    "source": "ogkalu_rtdetr",
                    "label": item["label_name"],
                    "license": "Apache-2.0 model card; source data requires due diligence",
                },
            }
        )

    bubbles = []
    for index, item in enumerate(_sort_reading_order(bubble_candidates), 1):
        bubbles.append(
            {
                "id": f"bubble_{index:03d}",
                "bbox": item["bbox"],
                "polygon": _bbox_to_polygon(item["bbox"]),
                "confidence": item["confidence"],
                "source_textline_ids": _textlines_inside(item["bbox"], textlines),
                "background_type": "unknown",
                "metadata": {
                    "source": "ogkalu_rtdetr",
                    "label": item["label_name"],
                    "license": "Apache-2.0 model card; source data requires due diligence",
                },
            }
        )

    raw_mask = _mask_from_boxes(text_candidates, width=width, height=height, dilation=2)
    bubble_mask = _mask_from_boxes(bubble_candidates, width=width, height=height, dilation=0)

    return {
        "schema_version": "ogkalu_rtdetr_adapter.v1",
        "provider": "ogkalu-rtdetr",
        "model_path": str(resolved_model),
        "regions": regions,
        "textlines": textlines,
        "bubbles": bubbles,
        "raw_mask": raw_mask,
        "bubble_mask": bubble_mask,
        "quality": {
            "model": "ogkalu/comic-text-and-bubble-detector detector-v4-s_int8.onnx",
            "score_threshold": score_threshold,
            "nms_threshold": nms_threshold,
            "candidate_count": len(candidates),
            "text_candidate_count": len(text_candidates),
            "bubble_candidate_count": len(bubble_candidates),
        },
    }


def _normalize_xyxy(box: np.ndarray | list[float], *, width: int, height: int) -> list[float] | None:
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
    ix1 = max(lx, rx)
    iy1 = max(ly, ry)
    ix2 = min(lx + lw, rx + rw)
    iy2 = min(ly + lh, ry + rh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = lw * lh + rw * rh - inter
    return inter / union if union > 0 else 0.0


def _dedupe(items: list[dict[str, Any]], *, nms_threshold: float, max_regions: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: (-value["confidence"], value["bbox"][1], value["bbox"][0])):
        if any(item["label"] == existing["label"] and _bbox_iou(item["bbox"], existing["bbox"]) > nms_threshold for existing in kept):
            continue
        kept.append(item)
        if len(kept) >= max_regions:
            break
    return kept


def _sort_reading_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _textlines_inside(bubble_bbox: list[float], textlines: list[dict[str, Any]]) -> list[str]:
    bx, by, bw, bh = bubble_bbox
    ids = []
    for line in textlines:
        lx, ly, lw, lh = line["bbox"]
        cx = lx + lw / 2
        cy = ly + lh / 2
        if bx <= cx <= bx + bw and by <= cy <= by + bh:
            ids.append(str(line["id"]))
    return ids


def _mask_from_boxes(items: list[dict[str, Any]], *, width: int, height: int, dilation: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in items:
        x, y, w, h = item["bbox"]
        x1 = int(round(x * width))
        y1 = int(round(y * height))
        x2 = int(round((x + w) * width))
        y2 = int(round((y + h) * height))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    if dilation > 0 and mask.any():
        kernel = np.ones((dilation * 2 + 1, dilation * 2 + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask
