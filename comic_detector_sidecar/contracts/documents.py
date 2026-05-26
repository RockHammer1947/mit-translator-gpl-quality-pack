from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, RootModel, field_validator

SCHEMA_VERSION = "comic-detector-sidecar.v1"
REGIONS_SCHEMA_VERSION = "comic_text_regions.v1"
TEXTLINES_SCHEMA_VERSION = "comic_text_lines.v1"
BUBBLES_SCHEMA_VERSION = "bubble_regions.v1"
LAYOUT_REGIONS_SCHEMA_VERSION = "layout_regions.v1"


class BBox(RootModel[list[float]]):
    @field_validator("root")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("BBox must contain exactly 4 values: [x, y, w, h]")
        for index, coordinate in enumerate(value):
            if not (0.0 <= coordinate <= 1.0):
                raise ValueError(f"BBox coordinate at index {index} must be between 0.0 and 1.0")
        x, y, w, h = value
        if x + w > 1.0001 or y + h > 1.0001:
            raise ValueError("BBox must fit within normalized image bounds")
        return value


class ComicTextRegion(BaseModel):
    id: str
    bbox: BBox
    confidence: float
    kind: Literal["text_region", "text_line", "speech_bubble"] = "text_region"
    reading_order: int
    provider: str
    polygon: list[list[float]] | None = None
    source_region_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        if len(value) < 4:
            raise ValueError("polygon must contain at least four points")
        for point in value:
            if len(point) != 2:
                raise ValueError("polygon point must be [x, y]")
            if not all(0.0 <= coordinate <= 1.0 for coordinate in point):
                raise ValueError("polygon coordinates must be normalized")
        return value


class BubbleRegion(BaseModel):
    id: str
    bbox: BBox
    polygon: list[list[float]] | None = None
    confidence: float
    source_textline_ids: list[str] = Field(default_factory=list)
    background_type: Literal["white", "black", "unknown"] = "unknown"
    mask_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        return ComicTextRegion.validate_polygon(value)


class LayoutRegion(BaseModel):
    id: str
    bbox: BBox
    polygon: list[list[float]] | None = None
    kind: Literal["text_cluster", "bubble", "narration", "sfx"] = "text_cluster"
    confidence: float
    source_textline_ids: list[str] = Field(default_factory=list)
    source_bubble_id: str | None = None
    render_priority: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        return ComicTextRegion.validate_polygon(value)


class ComicTextDetectionDocument(BaseModel):
    schema_version: str = REGIONS_SCHEMA_VERSION
    job_id: str
    provider: str
    source: dict[str, Any]
    regions: list[ComicTextRegion] = Field(default_factory=list)


class BubbleRegionsDocument(BaseModel):
    schema_version: str = BUBBLES_SCHEMA_VERSION
    job_id: str
    provider: str
    source: dict[str, Any]
    regions: list[BubbleRegion] = Field(default_factory=list)


class LayoutRegionsDocument(BaseModel):
    schema_version: str = LAYOUT_REGIONS_SCHEMA_VERSION
    job_id: str
    provider: str
    source: dict[str, Any]
    regions: list[LayoutRegion] = Field(default_factory=list)


class ComicDetectorManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    job_id: str
    status: Literal["completed", "failed", "cancelled"]
    provider: str
    source: Optional[dict[str, Any]] = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, Any]
    warnings: list[dict[str, str]] = Field(default_factory=list)


class DetectorOptions(BaseModel):
    model_path: Optional[Path] = None
    adapter_command: Optional[str] = None
    max_regions: int = 24
    min_confidence: float = 0.0
    detection_size: int = 2048
    text_threshold: float = 0.5
    box_threshold: float = 0.45
    unclip_ratio: float = 2.3
    coverage_preset: Literal["fast", "balanced", "quality"] = "balanced"
    nms_threshold: float = 0.35
    min_textline_area_ratio: float = 0.000015
