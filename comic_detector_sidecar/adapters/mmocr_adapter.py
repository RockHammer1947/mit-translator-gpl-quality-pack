from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_MMOCR_TEXTDET_MODEL = "dbnet_resnet18_fpnc_1200e_icdar2015"


def detect_mmocr(
    *,
    image_path: Path,
    model_name: str = DEFAULT_MMOCR_TEXTDET_MODEL,
    score_threshold: float,
    max_regions: int,
) -> dict[str, Any]:
    try:
        _assert_full_mmcv_ops()
        from mmocr.apis import TextDetInferencer
    except Exception as error:
        raise RuntimeError("mmocr with full mmcv ops is required for mmocr provider") from error

    if not image_path.exists():
        raise FileNotFoundError(f"input image not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to load image: {image_path}")
    height, width = image.shape[:2]

    # OpenMMLab may print model download/progress logs. Keep JSONL stdout clean.
    os.environ.setdefault("MMENGINE_HOME", str(Path.home() / ".cache" / "mmengine"))
    with contextlib.redirect_stdout(sys.stderr):
        inferencer = TextDetInferencer(model=model_name, device="cpu")
        result = inferencer(
            str(image_path),
            pred_score_thr=score_threshold,
            progress_bar=False,
            return_vis=False,
            save_vis=False,
            save_pred=False,
            print_result=False,
        )

    predictions = result.get("predictions") or []
    prediction = predictions[0] if predictions else {}
    polygons = prediction.get("polygons") or []
    bboxes = prediction.get("bboxes") or []
    scores = prediction.get("scores") or []
    candidates = _collect_candidates(
        polygons=polygons,
        bboxes=bboxes,
        scores=scores,
        width=width,
        height=height,
        score_threshold=score_threshold,
        max_regions=max_regions,
    )
    raw_mask = _mask_from_polygons(candidates, width=width, height=height)
    regions = []
    textlines = []
    for index, item in enumerate(candidates, 1):
        payload = {
            "id": f"line_{index:03d}",
            "bbox": item["bbox"],
            "confidence": item["confidence"],
            "reading_order": index,
            "polygon": item["polygon"],
            "metadata": {
                "source": "mmocr_textdet",
                "model": model_name,
                "license": "Apache-2.0 upstream OpenMMLab MMOCR; model license follows upstream model zoo",
            },
        }
        textlines.append({**payload, "kind": "text_line"})
        regions.append({**payload, "id": f"reg_{index:03d}", "kind": "text_region"})
    return {
        "schema_version": "mmocr_adapter.v1",
        "provider": "mmocr",
        "regions": regions,
        "textlines": textlines,
        "raw_mask": raw_mask,
        "quality": {
            "model": model_name,
            "score_threshold": score_threshold,
            "candidate_count": len(candidates),
        },
    }


def _assert_full_mmcv_ops() -> None:
    import mmcv  # noqa: F401

    try:
        import mmcv._ext  # noqa: F401
    except Exception as error:
        raise RuntimeError("full mmcv native ops are not installed; mmcv-lite is not enough for MMOCR") from error


def _collect_candidates(
    *,
    polygons: list[Any],
    bboxes: list[Any],
    scores: list[Any],
    width: int,
    height: int,
    score_threshold: float,
    max_regions: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    total = max(len(polygons), len(bboxes), len(scores))
    for index in range(total):
        score = float(scores[index]) if index < len(scores) else 1.0
        if score < score_threshold:
            continue
        polygon = _normalize_polygon(polygons[index], width=width, height=height) if index < len(polygons) else None
        bbox = _normalize_bbox(bboxes[index], width=width, height=height) if index < len(bboxes) else None
        if bbox is None and polygon is not None:
            bbox = _bbox_from_polygon(polygon)
        if polygon is None and bbox is not None:
            polygon = _bbox_to_polygon(bbox)
        if bbox is None or polygon is None:
            continue
        candidates.append({"bbox": bbox, "polygon": polygon, "confidence": round(score, 4)})
    return sorted(candidates, key=lambda item: (item["bbox"][1], item["bbox"][0]))[:max_regions]


def _normalize_polygon(points: Any, *, width: int, height: int) -> list[list[float]] | None:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if array.shape[0] < 4:
        return None
    normalized = []
    for x, y in array:
        normalized.append([round(float(np.clip(x, 0, width)) / width, 6), round(float(np.clip(y, 0, height)) / height, 6)])
    return normalized


def _normalize_bbox(values: Any, *, width: int, height: int) -> list[float] | None:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape[0] < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in array[:4]]
    x1 = max(0.0, min(width, x1))
    x2 = max(0.0, min(width, x2))
    y1 = max(0.0, min(height, y1))
    y2 = max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1 / width, 6), round(y1 / height, 6), round((x2 - x1) / width, 6), round((y2 - y1) / height, 6)]


def _bbox_from_polygon(polygon: list[list[float]]) -> list[float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    return [round(x1, 6), round(y1, 6), round(max(0.0, x2 - x1), 6), round(max(0.0, y2 - y1), 6)]


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _mask_from_polygons(items: list[dict[str, Any]], *, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in items:
        points = np.asarray([[round(x * width), round(y * height)] for x, y in item["polygon"]], dtype=np.int32)
        if points.shape[0] >= 3:
            cv2.fillPoly(mask, [points], 255)
    return mask
