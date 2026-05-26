from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass

import numpy as np

from manga_layout_sidecar import __version__
from manga_layout_sidecar.contracts import (
    BLOCKS_SCHEMA,
    SCHEMA_VERSION,
    MangaLayoutBlock,
    MangaLayoutBlocksDocument,
    MangaLayoutRequest,
)
from manga_layout_sidecar.geometry import (
    TextLineBox,
    merge_textline_boxes,
    normalize_polygon,
    polygon_from_bbox,
    polygon_to_normalized_bbox,
    union_bbox,
)


def merge_textlines(
    *,
    input_path: Path,
    output_dir: Path,
    manifest_path: Path | None,
    job_id: str,
    jsonl: bool,
    emit: Callable[[dict], None],
) -> dict[str, Any]:
    request = MangaLayoutRequest.model_validate_json(input_path.read_text(encoding="utf-8"))
    source = request.resolved_source()
    width = source.width or request.image_width or 1
    height = source.height or request.image_height or 1
    if width <= 0 or height <= 0:
        raise ValueError("image width/height must be positive for layout merge")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path or output_dir / "manga_layout_manifest.json"
    blocks_path = output_dir / "manga_layout_blocks.json"
    graph_path = output_dir / "layout_graph.json"
    quality_debug_path = output_dir / "layout_quality_debug.json"

    emit({"type": "progress", "job_id": job_id, "stage": "layout", "progress": 0.25, "message": "building textline geometry"})
    region_index = _build_region_index(request)
    textline_boxes = _to_textline_boxes(request, width, height, region_index)
    emit(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "layout",
            "progress": 0.55,
            "message": f"merging {len(textline_boxes)} textlines",
        }
    )
    groups, edges_debug = _merge_by_layout_regions(textline_boxes, width, height)
    groups = groups[: request.options.max_blocks]
    blocks = [_to_layout_block(group, index + 1, width, height) for index, group in enumerate(groups)]
    document = MangaLayoutBlocksDocument(
        job_id=job_id,
        source=source,
        options=request.options,
        blocks=blocks,
        metadata={
            "provider": "native_layout",
            "sidecar_version": __version__,
            "input_textline_count": len(request.textlines),
            "layout_block_count": len(blocks),
            "compatibility_modes": ["mit_textline_merge"],
        },
    )
    blocks_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")

    graph_doc = {
        "schema_version": "manga_layout_graph.v1",
        "job_id": job_id,
        "input_textline_count": len(request.textlines),
        "layout_block_count": len(blocks),
        "edges": edges_debug,
        "region_links": _region_links(blocks),
        "groups": [
            {
                "id": block.id,
                "source_textline_ids": block.source_textline_ids,
                "orientation": block.orientation,
                "bbox": block.bbox,
            }
            for block in blocks
        ],
    }
    graph_path.write_text(json.dumps(graph_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    quality_debug = _layout_quality_debug(request, blocks)
    quality_debug_path.write_text(json.dumps(quality_debug, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input_textline_count": len(request.textlines),
        "layout_block_count": len(blocks),
        "merged_group_count": sum(1 for block in blocks if len(block.source_textline_ids) > 1),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "status": "completed",
        "provider": "native_layout",
        "source": source.model_dump(),
        "artifacts": {
            "layout_blocks": str(blocks_path),
            "layout_graph": str(graph_path),
            "layout_region_graph": str(graph_path),
            "layout_quality_debug": str(quality_debug_path),
            "manifest": str(manifest_path),
        },
        "summary": summary,
        "warnings": [],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "document": document.model_dump(),
        "manifest": manifest,
        "artifacts": manifest["artifacts"],
    }


def _to_textline_boxes(
    request: MangaLayoutRequest,
    width: int,
    height: int,
    region_index: dict[str, dict[str, Any]],
) -> list[TextLineBox]:
    items = []
    for item in request.textlines:
        if item.confidence < request.options.min_confidence:
            continue
        if not request.options.preserve_punctuation and not item.text.strip():
            continue
        if item.polygon:
            pts = np.array([[point[0] * width, point[1] * height] for point in item.polygon[:4]], dtype=np.float32)
        else:
            pts = polygon_from_bbox(item.bbox, width, height)
        metadata = dict(item.metadata)
        assignment = _assign_region(item.id, item.bbox, metadata, region_index)
        layout_region = assignment.layout_region
        bubble_region = assignment.bubble_region
        source_region_id = item.source_region_id or metadata.get("source_region_id")
        source_region_bbox = item.source_region_bbox or _coerce_bbox(metadata.get("source_region_bbox"))
        layout_region_id = item.layout_region_id or (layout_region or {}).get("id")
        layout_region_bbox = item.layout_region_bbox or _coerce_bbox((layout_region or {}).get("bbox"))
        bubble_id = item.bubble_id or (bubble_region or {}).get("id")
        bubble_bbox = item.bubble_bbox or _coerce_bbox((bubble_region or {}).get("bbox"))
        metadata.update(
            {
                "layout_region_id": layout_region_id,
                "layout_region_bbox": layout_region_bbox,
                "bubble_id": bubble_id,
                "bubble_bbox": bubble_bbox,
            }
        )
        items.append(
            TextLineBox(
                points=pts,
                text=item.text,
                confidence=max(item.confidence, 1e-4),
                item_id=item.id,
                reading_order=item.reading_order,
                source_region_id=source_region_id,
                source_region_bbox=source_region_bbox,
                fg_color=item.foreground_color,
                bg_color=item.background_color,
                metadata=metadata,
            )
        )
    return items


def _merge_by_layout_regions(
    textline_boxes: list[TextLineBox],
    width: int,
    height: int,
) -> tuple[list[list[TextLineBox]], list[dict[str, Any]]]:
    partitions: dict[str, list[TextLineBox]] = {}
    for line in textline_boxes:
        key = str(line.metadata.get("layout_region_id") or line.metadata.get("bubble_id") or "__unassigned")
        partitions.setdefault(key, []).append(line)
    all_groups: list[list[TextLineBox]] = []
    all_edges: list[dict[str, Any]] = []
    for key, lines in partitions.items():
        if len(lines) == 1:
            all_groups.append(lines)
            continue
        groups, edges = merge_textline_boxes(lines, width, height)
        all_groups.extend(groups)
        for edge in edges:
            edge["layout_partition_id"] = key
        all_edges.extend(edges)
    all_groups.sort(key=lambda group: min(line.reading_order if line.reading_order is not None else 999999 for line in group))
    return all_groups, all_edges


def _to_layout_block(group: list[TextLineBox], index: int, width: int, height: int) -> MangaLayoutBlock:
    orientations = [line.orientation for line in group]
    majority = max(set(orientations), key=orientations.count) if orientations else "unknown"
    sorted_group = sorted(
        group,
        key=lambda line: (
            line.reading_order if line.reading_order is not None else 999999,
            -line.center[0] if majority == "v" else line.center[1],
        ),
    )
    texts = [line.text.strip() for line in sorted_group if line.text.strip()]
    separator = "" if majority == "v" else " "
    text = separator.join(texts)
    pts = np.concatenate([line.points for line in sorted_group], axis=0)
    bbox = polygon_to_normalized_bbox(pts, width, height)
    textline_bboxes = [polygon_to_normalized_bbox(line.points, width, height) for line in sorted_group]
    textline_polygons = [normalize_polygon(line.points, width, height) for line in sorted_group]
    source_region_bboxes = [line.source_region_bbox for line in sorted_group if line.source_region_bbox]
    layout_bboxes = [_coerce_bbox(line.metadata.get("layout_region_bbox")) for line in sorted_group]
    layout_bboxes = [item for item in layout_bboxes if item]
    bubble_bboxes = [_coerce_bbox(line.metadata.get("bubble_bbox")) for line in sorted_group]
    bubble_bboxes = [item for item in bubble_bboxes if item]
    layout_region_ids = sorted({str(line.metadata.get("layout_region_id")) for line in sorted_group if line.metadata.get("layout_region_id")})
    bubble_ids = sorted({str(line.metadata.get("bubble_id")) for line in sorted_group if line.metadata.get("bubble_id")})
    if len(layout_region_ids) == 1 and layout_bboxes:
        render_bbox = union_bbox(layout_bboxes)
    elif len(bubble_ids) == 1 and bubble_bboxes:
        render_bbox = union_bbox(bubble_bboxes)
    elif source_region_bboxes and len(set(line.source_region_id for line in sorted_group)) == 1:
        render_bbox = union_bbox(source_region_bboxes)
    else:
        render_bbox = bbox
    reading_orders = [line.reading_order for line in sorted_group if line.reading_order is not None]
    source_region_ids = sorted({line.source_region_id for line in sorted_group if line.source_region_id})
    return MangaLayoutBlock(
        id=f"blk_{index:03d}",
        text=text,
        bbox=bbox,
        confidence=float(sum(line.confidence for line in sorted_group) / max(1, len(sorted_group))),
        mask_bbox=union_bbox(textline_bboxes),
        render_bbox=render_bbox,
        polygon=normalize_polygon(sorted_group[0].points, width, height) if sorted_group else None,
        textline_bboxes=textline_bboxes,
        textline_polygons=textline_polygons,
        source_textline_ids=[line.item_id for line in sorted_group],
        source_region_ids=source_region_ids,
        bubble_id=bubble_ids[0] if len(bubble_ids) == 1 else None,
        layout_region_id=layout_region_ids[0] if len(layout_region_ids) == 1 else None,
        bubble_bbox=union_bbox(bubble_bboxes) if bubble_bboxes else None,
        layout_bbox=union_bbox(layout_bboxes) if layout_bboxes else None,
        foreground_color=next((line.fg_color for line in sorted_group if line.fg_color), None),
        background_color=next((line.bg_color for line in sorted_group if line.bg_color), None),
        orientation="vertical" if majority == "v" else "horizontal" if majority == "h" else "unknown",
        font_size=float(min(line.font_size for line in sorted_group)) if sorted_group else None,
        angle=float(np.mean([line.angle_degrees for line in sorted_group])) if sorted_group else None,
        reading_order=min(reading_orders) if reading_orders else index,
        metadata={
            "source": "native_layout",
            "source_textline_ids": [line.item_id for line in sorted_group],
            "source_region_ids": source_region_ids,
            "bubble_ids": bubble_ids,
            "layout_region_ids": layout_region_ids,
            "merged_textline_count": len(sorted_group),
            "original_metadata": [line.metadata for line in sorted_group],
        },
    )


@dataclass
class _RegionAssignment:
    layout_region: dict[str, Any] | None = None
    bubble_region: dict[str, Any] | None = None


def _build_region_index(request: MangaLayoutRequest) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for region in request.layout_regions:
        region_id = str(region.get("id") or "")
        if region_id:
            index[f"layout:{region_id}"] = region
        for textline_id in region.get("source_textline_ids") or []:
            index[f"textline-layout:{textline_id}"] = region
    for bubble in request.bubbles:
        bubble_id = str(bubble.get("id") or "")
        if bubble_id:
            index[f"bubble:{bubble_id}"] = bubble
        for textline_id in bubble.get("source_textline_ids") or []:
            index[f"textline-bubble:{textline_id}"] = bubble
    return index


def _assign_region(
    textline_id: str,
    bbox: list[float],
    metadata: dict[str, Any],
    region_index: dict[str, dict[str, Any]],
) -> _RegionAssignment:
    layout_region = region_index.get(f"textline-layout:{textline_id}")
    bubble_region = region_index.get(f"textline-bubble:{textline_id}")
    if layout_region is None and metadata.get("layout_region_id"):
        layout_region = region_index.get(f"layout:{metadata['layout_region_id']}")
    if bubble_region is None and metadata.get("bubble_id"):
        bubble_region = region_index.get(f"bubble:{metadata['bubble_id']}")
    if layout_region is None:
        layout_region = _best_region_by_center(bbox, [value for key, value in region_index.items() if key.startswith("layout:")])
    if bubble_region is None:
        bubble_region = _best_region_by_center(bbox, [value for key, value in region_index.items() if key.startswith("bubble:")])
    return _RegionAssignment(layout_region=layout_region, bubble_region=bubble_region)


def _best_region_by_center(bbox: list[float], regions: list[dict[str, Any]]) -> dict[str, Any] | None:
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    candidates = []
    for region in regions:
        region_bbox = _coerce_bbox(region.get("bbox"))
        if not region_bbox:
            continue
        rx, ry, rw, rh = region_bbox
        if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
            candidates.append((rw * rh, region))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _coerce_bbox(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    return None


def _region_links(blocks: list[MangaLayoutBlock]) -> list[dict[str, Any]]:
    return [
        {
            "block_id": block.id,
            "layout_region_id": block.layout_region_id,
            "bubble_id": block.bubble_id,
            "source_textline_ids": block.source_textline_ids,
        }
        for block in blocks
    ]


def _layout_quality_debug(request: MangaLayoutRequest, blocks: list[MangaLayoutBlock]) -> dict[str, Any]:
    input_ids = {item.id for item in request.textlines}
    output_ids = {textline_id for block in blocks for textline_id in block.source_textline_ids}
    return {
        "schema_version": "manga_layout_quality_debug.v1",
        "job_id": request.job_id,
        "input_textline_count": len(input_ids),
        "layout_block_count": len(blocks),
        "preserved_textline_count": len(input_ids & output_ids),
        "dropped_textline_ids": sorted(input_ids - output_ids),
        "merged_blocks": [
            {"block_id": block.id, "source_textline_ids": block.source_textline_ids}
            for block in blocks
            if len(block.source_textline_ids) > 1
        ],
    }
