from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from comic_detector_sidecar.contracts.documents import BBox, ComicTextRegion, DetectorOptions
from comic_detector_sidecar.model_manager import (
    bundled_adapter_command,
    default_comic_text_detector_model_path,
    status_for_comic_text_detector,
)
from comic_detector_sidecar.providers.base import ComicDetectionResult, ComicDetectorProvider


class ComicTextDetectorProvider(ComicDetectorProvider):
    name = "comic-text-detector"

    def is_available(self, options: DetectorOptions) -> bool:
        return status_for_comic_text_detector(options.model_path, _adapter_command(options)).available

    def doctor(self, options: DetectorOptions) -> dict:
        return status_for_comic_text_detector(options.model_path, _adapter_command(options)).to_doctor_payload()

    def detect_image(self, image_path: Path, options: DetectorOptions) -> ComicDetectionResult:
        command = _adapter_command(options)
        model_path = options.model_path or default_comic_text_detector_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"comic-text-detector model not found: {model_path}. Run `comic-detector-sidecar prepare-models`."
            )
        argv = shlex.split(command)
        if not shutil.which(argv[0]):
            raise FileNotFoundError(f"comic-text-detector adapter command not found: {argv[0]}")

        with tempfile.TemporaryDirectory(prefix="comic_text_detector_") as tmp:
            output_json = Path(tmp) / "comic_text_detector_output.json"
            raw_mask_path = Path(tmp) / "text_mask_raw.png"
            full_argv = [
                *argv,
                "--input",
                str(image_path),
                "--output",
                str(output_json),
                "--raw-mask-output",
                str(raw_mask_path),
                "--detection-size",
                str(options.detection_size),
                "--text-threshold",
                str(options.text_threshold),
                "--box-threshold",
                str(options.box_threshold),
                "--unclip-ratio",
                str(options.unclip_ratio),
                "--model-path",
                str(model_path),
            ]
            completed = subprocess.run(full_argv, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                stderr = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(f"comic-text-detector adapter failed with exit code {completed.returncode}: {stderr}")
            if not output_json.exists():
                raise FileNotFoundError(f"comic-text-detector adapter did not write output JSON: {output_json}")
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            raw_mask = cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE) if raw_mask_path.exists() else None
            return _parse_adapter_payload(payload, raw_mask)


def _adapter_command(options: DetectorOptions) -> str | None:
    return options.adapter_command or os.environ.get("COMIC_TEXT_DETECTOR_CMD") or bundled_adapter_command()


def _parse_adapter_payload(payload: dict, raw_mask: np.ndarray | None) -> ComicDetectionResult:
    regions = [
        _parse_region(item, index, "reg", "text_region")
        for index, item in enumerate(payload.get("regions") or payload.get("text_regions") or [], 1)
    ]
    textlines = [
        _parse_region(item, index, "line", "text_line")
        for index, item in enumerate(payload.get("textlines") or payload.get("text_lines") or [], 1)
    ]
    if not regions and textlines:
        regions = _regions_from_textlines(textlines)
    return ComicDetectionResult(regions=regions, textlines=textlines, raw_mask=raw_mask)


def _parse_region(item: dict, index: int, prefix: str, default_kind: str) -> ComicTextRegion:
    bbox = item.get("bbox")
    if not bbox and item.get("polygon"):
        bbox = _bbox_from_polygon(item["polygon"])
    if not bbox:
        raise ValueError(f"adapter item {index} did not contain bbox or polygon")
    return ComicTextRegion(
        id=str(item.get("id") or f"{prefix}_{index:03d}"),
        bbox=BBox([float(value) for value in bbox]),
        confidence=float(item.get("confidence", item.get("score", 0.0))),
        kind=item.get("kind") or default_kind,
        reading_order=int(item.get("reading_order", index)),
        provider="comic-text-detector",
        polygon=item.get("polygon") or _bbox_to_polygon(bbox),
        source_region_id=item.get("source_region_id"),
        metadata=item.get("metadata") or {},
    )


def _regions_from_textlines(textlines: list[ComicTextRegion]) -> list[ComicTextRegion]:
    return [
        ComicTextRegion(
            id=f"reg_{index:03d}",
            bbox=line.bbox,
            confidence=line.confidence,
            kind="text_region",
            reading_order=line.reading_order,
            provider=line.provider,
            polygon=line.polygon,
            metadata={"source": "textline_promoted_region"},
        )
        for index, line in enumerate(textlines, 1)
    ]


def _bbox_from_polygon(polygon: list[list[float]]) -> list[float]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [x1, y1, x2 - x1, y2 - y1]


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = [float(value) for value in bbox]
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
