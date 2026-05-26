from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "manga-cleaner-sidecar.v1"
CLEAN_REQUEST_SCHEMA = "manga_cleaner_request.v1"
CLEAN_BATCH_REQUEST_SCHEMA = "manga_cleaner_batch_request.v1"
CLEAN_BENCHMARK_REQUEST_SCHEMA = "manga_cleaner_benchmark_request.v1"
CLEAN_BENCHMARK_REPORT_SCHEMA = "manga_cleaner_benchmark_report.v1"
CLEAN_MANIFEST_SCHEMA = SCHEMA_VERSION


class CleanerError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class CleanerBlock(BaseModel):
    id: str
    text: str = ""
    translated_text: str | None = None
    bbox: list[float]
    confidence: float = 0.0
    source_region_id: str | None = None
    source_textline_id: str | None = None
    source_region_bbox: list[float] | None = None
    mask_bbox: list[float] | None = None
    render_bbox: list[float] | None = None
    textline_bboxes: list[list[float]] = Field(default_factory=list)
    polygon: list[list[float]] | None = None
    textline_polygons: list[list[list[float]]] = Field(default_factory=list)
    foreground_color: str | None = None
    background_color: str | None = None
    orientation: Literal["horizontal", "vertical", "unknown"] | None = None
    reading_order: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox", "source_region_bbox", "mask_bbox", "render_bbox")
    @classmethod
    def validate_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox must be [x, y, w, h]")
        x, y, w, h = [float(part) for part in value]
        if x < 0 or y < 0 or w < 0 or h < 0 or x > 1 or y > 1:
            raise ValueError("bbox coordinates must be normalized")
        return [x, y, min(w, 1 - x), min(h, 1 - y)]

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        return _validate_polygon(value)

    @field_validator("textline_polygons")
    @classmethod
    def validate_textline_polygons(cls, value: list[list[list[float]]]) -> list[list[list[float]]]:
        return [_validate_polygon(polygon) or [] for polygon in value]


class CleanerConfig(BaseModel):
    provider: Literal["lama-large-internal", "telea", "lama-large", "none"] = "telea"
    quality_preset: Literal["fast", "balanced", "quality"] = "balanced"
    mask_refine_mode: Literal["polygon_cc_refine", "mit_fit_text", "raw_mask", "threshold", "disabled"] = "polygon_cc_refine"
    mask_source: Literal["auto", "raw", "refined", "union", "threshold"] = "auto"
    mask_dilation_offset: int = Field(default=18, ge=0, le=96)
    kernel_size: int = Field(default=3, ge=1, le=31)
    component_min_area: int = Field(default=8, ge=1, le=10000)
    component_max_area_ratio: float = Field(default=0.35, ge=0.01, le=1.0)
    component_min_overlap_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    inpaint_radius: float = Field(default=3.0, ge=1.0, le=12.0)
    inpainting_size: int = Field(default=2048, ge=256, le=4096)
    model_path: Path | None = None
    lama_command: str | None = None
    device: str | None = None
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"


class CleanImageRequest(BaseModel):
    schema_version: str = CLEAN_REQUEST_SCHEMA
    job_id: str
    source_image: Path
    raw_mask_image: Path | None = None
    detector_refined_mask_image: Path | None = None
    mask_output: Path
    refined_mask_output: Path | None = None
    mask_debug_output: Path | None = None
    inpaint_quality_report_output: Path | None = None
    cleaned_output: Path
    manifest_output: Path
    config: CleanerConfig = Field(default_factory=CleanerConfig)
    blocks: list[CleanerBlock] = Field(default_factory=list)


class CleanerManifest(BaseModel):
    schema_version: str = CLEAN_MANIFEST_SCHEMA
    job_id: str
    status: Literal["completed", "failed", "cancelled"]
    provider: str
    source: dict[str, Any]
    artifacts: dict[str, str] = Field(default_factory=dict)
    model: dict[str, Any] | None = None
    summary: dict[str, Any]
    warnings: list[dict[str, str]] = Field(default_factory=list)


class CleanBatchItem(BaseModel):
    id: str
    source_image: Path
    blocks_path: Path | None = None
    raw_mask_image: Path | None = None
    detector_refined_mask_image: Path | None = None
    config: CleanerConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CleanBatchRequest(BaseModel):
    schema_version: str = CLEAN_BATCH_REQUEST_SCHEMA
    job_id: str
    config: CleanerConfig = Field(default_factory=CleanerConfig)
    items: list[CleanBatchItem] = Field(default_factory=list)


class CleanerBenchmarkGate(BaseModel):
    text_residual_ratio: float = 0.08
    text_residual_warn_ratio: float = 0.12
    background_damage_ratio: float = 0.08
    edge_halo_score: float = 0.15
    quality_score: float = 0.85


class CleanerBenchmarkItem(BaseModel):
    id: str
    source_image: Path
    blocks_path: Path | None = None
    raw_mask_image: Path | None = None
    detector_refined_mask_image: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CleanerBenchmarkRequest(BaseModel):
    schema_version: str = CLEAN_BENCHMARK_REQUEST_SCHEMA
    job_id: str
    provider: Literal["lama-large-internal", "telea", "lama-large", "none"] = "lama-large-internal"
    presets: list[Literal["fast", "balanced", "quality"]] = Field(default_factory=lambda: ["quality", "balanced"])
    gate: CleanerBenchmarkGate = Field(default_factory=CleanerBenchmarkGate)
    config: CleanerConfig | None = None
    items: list[CleanerBenchmarkItem] = Field(default_factory=list)


def _validate_polygon(value: list[list[float]] | None) -> list[list[float]] | None:
    if value is None:
        return None
    if len(value) < 4:
        raise ValueError("polygon must contain at least four points")
    normalized: list[list[float]] = []
    for point in value:
        if len(point) != 2:
            raise ValueError("polygon point must be [x, y]")
        x, y = [float(part) for part in point]
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("polygon coordinates must be normalized")
        normalized.append([x, y])
    return normalized
