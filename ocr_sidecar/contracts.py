from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, RootModel, field_validator

SCHEMA_VERSION = "ocr-sidecar.v1"
BLOCKS_SCHEMA_VERSION = "ocr_blocks.v1"
BATCH_REQUEST_SCHEMA_VERSION = "ocr_batch_request.v1"
BATCH_RESULT_SCHEMA_VERSION = "ocr_batch_result.v1"


class BBox(RootModel[list[float]]):
    @field_validator("root")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("BBox must contain exactly 4 values: [x, y, w, h]")
        for index, coordinate in enumerate(value):
            if not (0.0 <= coordinate <= 1.0):
                raise ValueError(
                    f"BBox coordinate at index {index} ({coordinate}) must be between 0.0 and 1.0"
                )
        return value


class Polygon(RootModel[list[list[float]]]):
    @field_validator("root")
    @classmethod
    def validate_polygon(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) < 4:
            raise ValueError("Polygon must contain at least four points")
        for point in value:
            if len(point) != 2:
                raise ValueError("Polygon point must be [x, y]")
            if not all(0.0 <= float(coordinate) <= 1.0 for coordinate in point):
                raise ValueError("Polygon coordinates must be normalized")
        return value


class OcrTextBlock(BaseModel):
    text: str
    confidence: float
    bbox: BBox
    engine: str
    polygon: Optional[Polygon] = None
    textline_polygons: list[Polygon] = Field(default_factory=list)
    foreground_color: Optional[str] = None
    background_color: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OcrArtifactBlock(BaseModel):
    id: str
    text: str
    confidence: float
    bbox: BBox
    polygon: Optional[Polygon] = None
    textline_polygons: list[Polygon] = Field(default_factory=list)
    foreground_color: Optional[str] = None
    background_color: Optional[str] = None
    reading_order: int
    engine: str
    language_hint: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class OcrLayoutRegion(BaseModel):
    id: str
    bbox: BBox
    polygon: Optional[Polygon] = None
    confidence: float = 1.0
    kind: Optional[str] = None
    source_textline_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchOptions(BaseModel):
    quality_preset: Literal["fast", "balanced", "quality"] = "balanced"
    retry_profile: Literal["none", "mit_quality"] = "none"
    preserve_empty_textlines: bool = False
    debug_level: Literal["none", "crops", "attempts"] = "none"
    use_mocr_merge: bool = False
    mocr_merge_mode: Literal["off", "debug", "fill_empty"] = "off"
    region_hint_min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    region_crop_padding: float = Field(default=0.015, ge=0.0, le=0.2)


class BatchItem(BaseModel):
    id: str
    image_path: Path
    region: Optional[BBox] = None
    polygon: Optional[Polygon] = None
    crop_policy: Optional[Literal["bbox", "polygon_perspective", "adaptive", "mit_textline"]] = None
    target_text_height: Optional[int] = Field(default=None, ge=8, le=256)
    padding_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    preprocess: Optional[Literal["none", "auto", "manga_enhance"]] = None
    probability_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    retry_profile: Optional[Literal["none", "mit_quality"]] = None
    preserve_placeholder: Optional[bool] = None
    lang_hint: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", mode="before")
    @classmethod
    def parse_id(cls, value: Any) -> str:
        return str(value)


class BatchRequest(BaseModel):
    schema_version: str = BATCH_REQUEST_SCHEMA_VERSION
    job_id: Optional[str] = None
    engine: str
    lang_hint: str = "auto"
    language_hint: Optional[str] = None
    options: BatchOptions = Field(default_factory=BatchOptions)
    items: list[BatchItem] = Field(default_factory=list)
    bubbles: list[OcrLayoutRegion] = Field(default_factory=list)
    layout_regions: list[OcrLayoutRegion] = Field(default_factory=list)

    def effective_language_hint(self) -> str:
        return self.language_hint or self.lang_hint


class BatchResultItem(BaseModel):
    id: str
    image_path: str
    blocks: list[OcrArtifactBlock] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: Literal["recognized", "empty", "failed", "placeholder"] = "recognized"
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    selected_attempt_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class BatchResultDocument(BaseModel):
    schema_version: str = BATCH_RESULT_SCHEMA_VERSION
    job_id: str
    engine: str
    language_hint: str
    items: list[BatchResultItem] = Field(default_factory=list)


class OcrBlocksDocument(BaseModel):
    schema_version: str = BLOCKS_SCHEMA_VERSION
    job_id: str
    engine: str
    language_hint: str
    source: dict[str, Any]
    blocks: list[OcrArtifactBlock] = Field(default_factory=list)


class OcrManifestImage(BaseModel):
    image_id: str
    image_path: str
    blocks_path: Optional[str] = None
    block_count: int
    avg_confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ManifestWarning(BaseModel):
    code: str
    message: str


class OcrManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    job_id: str
    status: Literal["completed", "failed", "cancelled"]
    engine: str
    language_hint: str
    source: Optional[dict[str, Any]] = None
    images: list[OcrManifestImage] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, Any]
    warnings: list[ManifestWarning] = Field(default_factory=list)
