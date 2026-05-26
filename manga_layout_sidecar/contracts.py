from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "manga-layout-sidecar.v1"
REQUEST_SCHEMA = "manga_layout_request.v1"
BLOCKS_SCHEMA = "manga_layout_blocks.v1"


BBox = list[float]
Point = list[float]
Polygon = list[Point]


class MangaLayoutSource(BaseModel):
    type: Literal["image"] = "image"
    path: str
    width: int | None = None
    height: int | None = None


class MangaLayoutOptions(BaseModel):
    mode: Literal["native_layout", "mit_textline_merge"] = "native_layout"
    max_blocks: int = Field(default=64, ge=1, le=512)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    preserve_punctuation: bool = True


class MangaLayoutTextLine(BaseModel):
    id: str
    text: str = ""
    bbox: BBox
    polygon: Polygon | None = None
    confidence: float = 0.0
    reading_order: int | None = None
    source_region_id: str | None = None
    source_region_bbox: BBox | None = None
    bubble_id: str | None = None
    bubble_bbox: BBox | None = None
    layout_region_id: str | None = None
    layout_region_bbox: BBox | None = None
    foreground_color: str | None = None
    background_color: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox", "source_region_bbox", "bubble_bbox", "layout_region_bbox")
    @classmethod
    def validate_bbox(cls, value: BBox | None) -> BBox | None:
        if value is not None and len(value) != 4:
            raise ValueError("bbox must be [x, y, w, h]")
        return value

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: Polygon | None) -> Polygon | None:
        if value is not None and len(value) < 4:
            raise ValueError("polygon must contain at least four points")
        return value


class MangaLayoutRequest(BaseModel):
    schema_version: str = REQUEST_SCHEMA
    job_id: str | None = None
    source: MangaLayoutSource | None = None
    source_image: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    options: MangaLayoutOptions = Field(default_factory=MangaLayoutOptions)
    textlines: list[MangaLayoutTextLine] = Field(default_factory=list)
    regions: list[dict[str, Any]] = Field(default_factory=list)
    bubbles: list[dict[str, Any]] = Field(default_factory=list)
    layout_regions: list[dict[str, Any]] = Field(default_factory=list)
    text_mask_path: str | None = None

    def resolved_source(self) -> MangaLayoutSource:
        if self.source is not None:
            return self.source
        return MangaLayoutSource(
            path=self.source_image or "",
            width=self.image_width,
            height=self.image_height,
        )


class MangaLayoutBlock(BaseModel):
    id: str
    text: str
    bbox: BBox
    confidence: float
    mask_bbox: BBox
    render_bbox: BBox
    polygon: Polygon | None = None
    textline_bboxes: list[BBox] = Field(default_factory=list)
    textline_polygons: list[Polygon] = Field(default_factory=list)
    source_textline_ids: list[str] = Field(default_factory=list)
    source_region_ids: list[str] = Field(default_factory=list)
    bubble_id: str | None = None
    layout_region_id: str | None = None
    bubble_bbox: BBox | None = None
    layout_bbox: BBox | None = None
    foreground_color: str | None = None
    background_color: str | None = None
    orientation: Literal["horizontal", "vertical", "unknown"] = "unknown"
    font_size: float | None = None
    angle: float | None = None
    reading_order: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MangaLayoutBlocksDocument(BaseModel):
    schema_version: str = BLOCKS_SCHEMA
    job_id: str
    source: MangaLayoutSource
    options: MangaLayoutOptions
    blocks: list[MangaLayoutBlock]
    metadata: dict[str, Any] = Field(default_factory=dict)
