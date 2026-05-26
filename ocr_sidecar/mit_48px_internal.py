from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ocr_sidecar.model_manager import engine_model_dir, resolve_mit_48px_internal_runtime


async def recognize(payload: dict[str, Any]) -> dict[str, Any]:
    mit_repo = resolve_mit_48px_internal_runtime()
    if mit_repo is None:
        raise RuntimeError("manga-image-translator runtime not found. Set OCR_MIT_REPO.")
    _install_mit_namespace(mit_repo)
    with redirect_stdout(sys.stderr):
        from manga_translator.config import OcrConfig
        from manga_translator.ocr.model_48px import Model48pxOCR
        from manga_translator.utils import ModelWrapper, Quadrilateral

    model_dir = engine_model_dir("mit-48px")
    ModelWrapper._MODEL_DIR = str(model_dir)
    Model48pxOCR._MODEL_DIR = str(model_dir)
    Model48pxOCR._MODEL_SUB_DIR = ""

    image_path = Path(str(payload["image_path"]))
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"failed to load image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    orientation = _payload_orientation(payload)
    device = str(payload.get("device") or os.environ.get("OCR_MIT_48PX_DEVICE") or _default_device())
    ocr = Model48pxOCR()
    with redirect_stdout(sys.stderr):
        await ocr.load(device)

    thresholds = _payload_thresholds(payload)
    attempts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds, 1):
        attempt_id = str((payload.get("metadata") or {}).get("attempt_id") or f"mit48_{index}")
        if len(thresholds) > 1:
            attempt_id = f"mit48_{index}"
        quad = Quadrilateral(
            _payload_polygon_to_pixels(payload, width=width, height=height),
            "",
            1.0,
        )
        if orientation == "vertical":
            quad.direction = "v"
        elif orientation == "horizontal":
            quad.direction = "h"
        debug_crop_path = _write_debug_crop(payload, image, quad, orientation, threshold, attempt_id)
        config = OcrConfig(prob=float(threshold) if threshold is not None else None)
        with redirect_stdout(sys.stderr):
            results = await ocr.recognize(image, [quad], config, verbose=False)
        attempt_blocks = [
            _result_to_block(
                result,
                payload=payload,
                width=width,
                height=height,
                mit_repo=mit_repo,
                model_dir=model_dir,
                device=device,
                attempt_id=attempt_id,
                threshold=threshold,
                debug_crop_path=debug_crop_path,
            )
            for result in results
        ]
        raw_texts = [str(block.get("text", "")) for block in attempt_blocks]
        attempt = {
            "attempt_id": attempt_id,
            "recognizer": "mit-48px-internal",
            "probability_threshold": threshold,
            "status": "recognized" if any(text.strip() for text in raw_texts) else "empty",
            "texts": raw_texts,
            "max_confidence": max([float(block.get("confidence", 0.0) or 0.0) for block in attempt_blocks], default=0.0),
            **({"debug_crop_path": str(debug_crop_path)} if debug_crop_path else {}),
        }
        attempts.append(attempt)
        candidates.extend(block for block in attempt_blocks if str(block.get("text", "")).strip())

    blocks = _select_blocks(candidates)
    selected_attempt_id = blocks[0]["metadata"].get("attempt_id") if blocks else None
    return {"blocks": blocks, "attempts": attempts, "selected_attempt_id": selected_attempt_id}


def _payload_thresholds(payload: dict[str, Any]) -> list[float | None]:
    raw = payload.get("probability_thresholds")
    if isinstance(raw, list) and raw:
        values: list[float | None] = []
        for item in raw:
            values.append(None if item is None else float(item))
        return values
    return [None if payload.get("probability_threshold") is None else float(payload["probability_threshold"])]


def _result_to_block(
    result: Any,
    *,
    payload: dict[str, Any],
    width: int,
    height: int,
    mit_repo: Path,
    model_dir: Path,
    device: str,
    attempt_id: str,
    threshold: float | None,
    debug_crop_path: Path | None,
) -> dict[str, Any]:
    text = str(getattr(result, "text", "") or "").strip()
    points = np.array(getattr(result, "pts"), dtype=np.float32)
    bbox = _normalized_bbox(points, width=width, height=height)
    polygon = _normalized_polygon(points, width=width, height=height)
    confidence = float(getattr(result, "prob", 0.0) or 0.0)
    quality = _text_quality_score(text, confidence)
    return {
        "text": text,
        "confidence": confidence,
        "bbox": bbox,
        "polygon": polygon,
        "textline_polygons": [polygon],
        "foreground_color": _hex_color(
            getattr(result, "fg_r", 0),
            getattr(result, "fg_g", 0),
            getattr(result, "fg_b", 0),
        ),
        "background_color": _hex_color(
            getattr(result, "bg_r", 255),
            getattr(result, "bg_g", 255),
            getattr(result, "bg_b", 255),
        ),
        "metadata": {
            **(payload.get("metadata") or {}),
            "provider": "mit-48px-internal",
            "mit_repo": str(mit_repo),
            "model_dir": str(model_dir),
            "device": device,
            "assigned_direction": getattr(result, "assigned_direction", None),
            "attempt_id": attempt_id,
            "recognizer": "mit-48px-internal",
            "probability_threshold": threshold,
            "text_quality_score": quality,
            **({"debug_crop_path": str(debug_crop_path)} if debug_crop_path else {}),
        },
    }


def _select_blocks(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [
        (float(candidate.get("metadata", {}).get("text_quality_score", 0.0) or 0.0), candidate)
        for candidate in candidates
    ]
    scored = [(score, candidate) for score, candidate in scored if _candidate_is_usable(candidate, score)]
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    return [scored[0][1]]


def _candidate_is_usable(candidate: dict[str, Any], score: float) -> bool:
    text = str(candidate.get("text", "")).strip()
    if not text:
        return False
    confidence = float(candidate.get("confidence", 0.0) or 0.0)
    japanese = sum(1 for char in text if "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff")
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    if japanese > 0:
        return score > 0.2
    if latin >= 3:
        return confidence >= 0.08 or score > 1.5
    return score > 1.8


def _text_quality_score(text: str, confidence: float) -> float:
    cleaned = text.strip()
    if not cleaned:
        return 0.0
    japanese = sum(1 for char in cleaned if "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff")
    latin = sum(1 for char in cleaned if char.isascii() and char.isalpha())
    digits = sum(1 for char in cleaned if char.isascii() and char.isdigit())
    punctuation = sum(1 for char in cleaned if char in "!?！？…。、，．・「」『』ー〜～")
    symbols = len(cleaned) - japanese - latin - digits - punctuation
    score = confidence * 2.0 + japanese * 2.0 + min(latin, 8) * 0.45 + punctuation * 0.2 + min(len(cleaned), 12) * 0.08
    if len(cleaned) <= 1:
        score -= 0.8
    if symbols > 0:
        score -= symbols * 0.7
    if japanese == 0 and latin < 3 and punctuation > 0:
        score -= 1.0
    return float(score)


def _write_debug_crop(
    payload: dict[str, Any],
    image: np.ndarray,
    quad: Any,
    orientation: str | None,
    threshold: float | None,
    attempt_id: str,
) -> Path | None:
    debug_dir = payload.get("debug_output_dir")
    if not debug_dir:
        return None
    try:
        target_height = int(payload.get("target_text_height") or 48)
        direction = "v" if orientation == "vertical" else "h" if orientation == "horizontal" else getattr(quad, "direction", "h")
        crop = quad.get_transformed_region(image, direction, target_height)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        item_id = str(metadata.get("ocr_batch_item_id") or metadata.get("source_textline_id") or "img")
        safe_id = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in f"{item_id}_{attempt_id}")
        path = Path(str(debug_dir)) / f"{safe_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img_data = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        if direction == "v":
            img_data = cv2.rotate(img_data, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(path), img_data)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": "ocr_crop_debug.v1",
                    "crop_image": str(path),
                    "attempt_id": attempt_id,
                    "recognizer": "mit-48px-internal",
                    "probability_threshold": threshold,
                    "direction": direction,
                    "target_text_height": target_height,
                    "metadata": metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path
    except Exception as error:
        print(f"failed to write OCR debug crop: {error}", file=sys.stderr)
        return None


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        response = asyncio.run(recognize(payload))
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
        sys.stdout.flush()
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


def _payload_polygon_to_pixels(payload: dict[str, Any], *, width: int, height: int) -> np.ndarray:
    polygon = payload.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 4:
        region = payload.get("region") or [0.0, 0.0, 1.0, 1.0]
        x, y, w, h = [float(value) for value in region]
        polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    points = []
    for x, y in polygon[:4]:
        points.append([float(x) * width, float(y) * height])
    return np.array(points, dtype=np.float32)


def _install_mit_namespace(mit_repo: Path) -> None:
    package_root = mit_repo / "manga_translator"
    existing = sys.modules.get("manga_translator")
    if existing is None or not getattr(existing, "__path__", None):
        package = types.ModuleType("manga_translator")
        package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        sys.modules["manga_translator"] = package
    existing_ocr = sys.modules.get("manga_translator.ocr")
    if existing_ocr is None or not getattr(existing_ocr, "__path__", None):
        ocr_package = types.ModuleType("manga_translator.ocr")
        ocr_package.__path__ = [str(package_root / "ocr")]  # type: ignore[attr-defined]
        sys.modules["manga_translator.ocr"] = ocr_package
    if str(mit_repo) not in sys.path:
        sys.path.insert(0, str(mit_repo))


def _payload_orientation(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    value = metadata.get("orientation") or metadata.get("textline_orientation")
    if isinstance(value, str) and value:
        return value.lower()
    region = payload.get("region")
    if isinstance(region, list) and len(region) == 4:
        _, _, w, h = [float(item) for item in region]
        return "vertical" if h > w * 1.15 else "horizontal"
    return None


def _normalized_bbox(points: np.ndarray, *, width: int, height: int) -> list[float]:
    left = float(np.min(points[:, 0]) / width)
    top = float(np.min(points[:, 1]) / height)
    right = float(np.max(points[:, 0]) / width)
    bottom = float(np.max(points[:, 1]) / height)
    return [_clamp01(left), _clamp01(top), _clamp01(right - left), _clamp01(bottom - top)]


def _normalized_polygon(points: np.ndarray, *, width: int, height: int) -> list[list[float]]:
    return [[_clamp01(float(x) / width), _clamp01(float(y) / height)] for x, y in points.tolist()]


def _hex_color(r: Any, g: Any, b: Any) -> str:
    return f"#{_byte(r):02x}{_byte(g):02x}{_byte(b):02x}"


def _byte(value: Any) -> int:
    return max(0, min(255, int(round(float(value)))))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _default_device() -> str:
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


if __name__ == "__main__":
    main()
