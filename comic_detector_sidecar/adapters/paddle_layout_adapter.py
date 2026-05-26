from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TEXT_LABELS = {"text", "vertical_text", "paragraph_title", "doc_title", "aside_text", "content"}


def detect_paddle_layout(
    *,
    image_path: Path,
    score_threshold: float,
    max_regions: int,
) -> dict[str, Any]:
    try:
        import paddlex
    except Exception as error:
        raise RuntimeError("paddlex and paddlepaddle are required for paddle-layout provider") from error
    if not image_path.exists():
        raise FileNotFoundError(f"input image not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to load image: {image_path}")
    height, width = image.shape[:2]

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    with contextlib.redirect_stdout(sys.stderr):
        model = paddlex.create_model("PP-DocLayoutV3")
        results = list(model.predict(str(image_path)))
    boxes = []
    for result in results:
        payload = getattr(result, "json", None)
        if isinstance(payload, dict):
            items = payload.get("res", {}).get("boxes", [])
        elif isinstance(result, dict):
            items = result.get("boxes", [])
        else:
            items = []
        boxes.extend(items)

    candidates = []
    for item in boxes:
        label = str(item.get("label", "unknown"))
        score = float(item.get("score", 0.0))
        if label not in TEXT_LABELS or score < score_threshold:
            continue
        coordinate = item.get("coordinate")
        if not coordinate or len(coordinate) != 4:
            continue
        bbox = _normalize_xyxy(coordinate, width=width, height=height)
        if bbox is None:
            continue
        polygon = item.get("polygon_points")
        candidates.append(
            {
                "bbox": bbox,
                "polygon": _normalize_polygon(polygon, width=width, height=height) if polygon is not None else _bbox_to_polygon(bbox),
                "confidence": round(score, 4),
                "label": label,
                "order": int(item.get("order", len(candidates) + 1)),
            }
        )
    candidates = sorted(candidates, key=lambda item: (item["order"], item["bbox"][1], item["bbox"][0]))[:max_regions]
    textlines = []
    for index, item in enumerate(candidates, 1):
        textlines.append(
            {
                "id": f"line_{index:03d}",
                "bbox": item["bbox"],
                "confidence": item["confidence"],
                "kind": "text_line",
                "reading_order": index,
                "polygon": item["polygon"],
                "metadata": {
                    "source": "paddle_pp_doclayout_v3",
                    "label": item["label"],
                    "license": "Apache-2.0 upstream PaddleOCR/PaddleX",
                },
            }
        )
    regions = []
    for index, item in enumerate(candidates, 1):
        regions.append(
            {
                "id": f"reg_{index:03d}",
                "bbox": item["bbox"],
                "confidence": item["confidence"],
                "kind": "text_region",
                "reading_order": index,
                "polygon": item["polygon"],
                "metadata": {
                    "source": "paddle_pp_doclayout_v3",
                    "label": item["label"],
                    "license": "Apache-2.0 upstream PaddleOCR/PaddleX",
                },
            }
        )
    raw_mask = _mask_from_boxes(candidates, width=width, height=height)
    return {
        "schema_version": "paddle_layout_adapter.v1",
        "provider": "paddle-layout",
        "regions": regions,
        "textlines": textlines,
        "raw_mask": raw_mask,
        "quality": {
            "model": "PP-DocLayoutV3",
            "score_threshold": score_threshold,
            "candidate_count": len(candidates),
            "label_counts": _label_counts(candidates),
        },
    }


def _normalize_xyxy(box: list[float], *, width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(width, x1))
    x2 = max(0.0, min(width, x2))
    y1 = max(0.0, min(height, y1))
    y2 = max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1 / width, 6), round(y1 / height, 6), round((x2 - x1) / width, 6), round((y2 - y1) / height, 6)]


def _normalize_polygon(points: Any, *, width: int, height: int) -> list[list[float]]:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return [[round(float(x) / width, 6), round(float(y) / height, 6)] for x, y in array[:4]]


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


def _label_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        label = str(item.get("label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return counts
