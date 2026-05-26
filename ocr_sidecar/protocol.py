import json
import platform
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

import typer

from ocr_sidecar import __version__
from ocr_sidecar.contracts import (
    BLOCKS_SCHEMA_VERSION,
    SCHEMA_VERSION,
    BatchResultDocument,
    BBox,
    ManifestWarning,
    OcrArtifactBlock,
    OcrBlocksDocument,
    OcrManifest,
    OcrTextBlock,
)

ERROR_CODES = {
    "INPUT_NOT_FOUND",
    "INVALID_REGION",
    "OCR_ENGINE_NOT_FOUND",
    "OCR_ENGINE_LOAD_FAILED",
    "OCR_EXECUTION_FAILED",
    "INVALID_BATCH_REQUEST",
    "OUTPUT_WRITE_FAILED",
    "MANIFEST_WRITE_FAILED",
    "OCR_CANCELLED",
}

WARNING_CODES = {
    "LOW_OCR_CONFIDENCE",
    "EMPTY_OCR_RESULT",
    "PARTIAL_BATCH_FAILURE",
    "PADDLE_NOT_INSTALLED",
    "REGION_OUT_OF_BOUNDS",
}


class SidecarError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def make_job_id(prefix: str = "job_ocr") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def emit_event(event: dict[str, Any], jsonl: bool) -> None:
    if not jsonl:
        return
    event.setdefault("schema_version", SCHEMA_VERSION)
    typer.echo(json.dumps(event, ensure_ascii=False, separators=(",", ":")))


def dump_json(path: Path, payload: dict[str, Any]) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except OSError as e:
        raise SidecarError("OUTPUT_WRITE_FAILED", f"Failed to write {path}: {e}") from e


def dump_manifest(path: Path, manifest: OcrManifest) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
    except OSError as e:
        raise SidecarError("MANIFEST_WRITE_FAILED", f"Failed to write manifest {path}: {e}") from e


def to_artifact_blocks(
    blocks: list[OcrTextBlock],
    engine: str,
    language_hint: str,
) -> list[OcrArtifactBlock]:
    sorted_blocks = sorted(blocks, key=lambda block: (block.bbox.root[1], block.bbox.root[0]))
    return [
        OcrArtifactBlock(
            id=f"blk_{idx:03d}",
            text=block.text,
            confidence=block.confidence,
            bbox=block.bbox,
            polygon=block.polygon,
            textline_polygons=block.textline_polygons,
            foreground_color=block.foreground_color,
            background_color=block.background_color,
            reading_order=idx,
            engine=block.engine or engine,
            language_hint=language_hint,
            metadata=block.metadata,
        )
        for idx, block in enumerate(sorted_blocks, 1)
    ]


def avg_confidence(blocks: list[OcrArtifactBlock]) -> float:
    if not blocks:
        return 0.0
    return float(sum(block.confidence for block in blocks) / len(blocks))


def build_blocks_document(
    job_id: str,
    engine: str,
    language_hint: str,
    source: dict[str, Any],
    blocks: list[OcrArtifactBlock],
) -> OcrBlocksDocument:
    return OcrBlocksDocument(
        schema_version=BLOCKS_SCHEMA_VERSION,
        job_id=job_id,
        engine=engine,
        language_hint=language_hint,
        source=source,
        blocks=blocks,
    )


def build_summary(image_count: int, block_count: int, confidences: list[float]) -> dict[str, Any]:
    empty_image_count = image_count - len([confidence for confidence in confidences if confidence > 0.0])
    return {
        "image_count": image_count,
        "block_count": block_count,
        "avg_confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
        "empty_image_count": empty_image_count,
    }


def warning_for_blocks(blocks: list[OcrArtifactBlock], min_confidence: float) -> Optional[ManifestWarning]:
    if not blocks:
        return ManifestWarning(code="EMPTY_OCR_RESULT", message="OCR produced no text blocks.")
    confidence = avg_confidence(blocks)
    if confidence < min_confidence:
        return ManifestWarning(
            code="LOW_OCR_CONFIDENCE",
            message=f"Average OCR confidence is low ({confidence:.2f}).",
        )
    return None


def doctor_payload() -> dict[str, Any]:
    from ocr_sidecar.model_manager import all_engine_statuses

    engines = [status.to_doctor_payload() for status in all_engine_statuses()]
    return {
        "type": "doctor",
        "job_id": "doctor",
        "schema_version": SCHEMA_VERSION,
        "sidecar_version": __version__,
        "python_version": platform.python_version(),
        "engines": engines,
    }


def error_to_code(error: Exception) -> str:
    if isinstance(error, SidecarError):
        return error.code
    message = str(error).lower()
    if "not installed" in message or "initialize" in message:
        return "OCR_ENGINE_LOAD_FAILED"
    if "not supported" in message:
        return "OCR_ENGINE_NOT_FOUND"
    if "region" in message:
        return "INVALID_REGION"
    if "batch" in message or "json" in message:
        return "INVALID_BATCH_REQUEST"
    return "OCR_EXECUTION_FAILED"


def print_error(error: Exception, job_id: str, jsonl: bool) -> None:
    code = error_to_code(error)
    payload = {
        "type": "error",
        "job_id": job_id,
        "schema_version": SCHEMA_VERSION,
        "code": code,
        "message": str(error),
    }
    if jsonl:
        typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        typer.echo(
            json.dumps({"error": {"code": code, "message": str(error)}}, ensure_ascii=False),
            err=True,
        )


def parse_region(raw: Optional[str]) -> Optional[BBox]:
    if raw is None:
        return None
    try:
        return BBox([float(part.strip()) for part in raw.split(",")])
    except Exception as e:
        raise SidecarError("INVALID_REGION", f"Invalid --region value: {raw}") from e


def read_json(path: Optional[Path], read_stdin: bool) -> dict[str, Any]:
    if path is not None and read_stdin:
        raise SidecarError("INVALID_BATCH_REQUEST", "Use either --input or --stdin, not both.")
    if path is None and not read_stdin:
        raise SidecarError("INVALID_BATCH_REQUEST", "Missing batch input. Use --input or --stdin.")
    raw = sys.stdin.read() if read_stdin else path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SidecarError("INVALID_BATCH_REQUEST", "Batch input is not valid JSON.") from e
    if not isinstance(payload, dict):
        raise SidecarError("INVALID_BATCH_REQUEST", "Batch input must be a JSON object.")
    return payload


def legacy_blocks_response(engine: str, blocks: list[OcrArtifactBlock]) -> dict[str, Any]:
    return {
        "engine": engine,
        "blocks": [
            {
                "text": block.text,
                "confidence": block.confidence,
                "bbox": block.bbox.root,
                "engine": block.engine,
            }
            for block in blocks
        ],
    }


def legacy_batch_response(result: BatchResultDocument) -> dict[str, Any]:
    return {
        "engine": result.engine,
        "items": [
            {
                "id": item.id,
                "blocks": [
                    {
                        "text": block.text,
                        "confidence": block.confidence,
                        "bbox": block.bbox.root,
                        "engine": block.engine,
                    }
                    for block in item.blocks
                ],
                **({"error_code": item.error_code} if item.error_code else {}),
                **({"error_message": item.error_message} if item.error_message else {}),
            }
            for item in result.items
        ],
    }
