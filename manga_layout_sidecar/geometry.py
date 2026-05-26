from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class TextLineBox:
    points: np.ndarray
    text: str
    confidence: float
    item_id: str
    reading_order: int | None = None
    source_region_id: str | None = None
    source_region_bbox: list[float] | None = None
    fg_color: str | None = None
    bg_color: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def min_x(self) -> float:
        return float(np.min(self.points[:, 0]))

    @property
    def min_y(self) -> float:
        return float(np.min(self.points[:, 1]))

    @property
    def max_x(self) -> float:
        return float(np.max(self.points[:, 0]))

    @property
    def max_y(self) -> float:
        return float(np.max(self.points[:, 1]))

    @property
    def width(self) -> float:
        return max(1.0, self.max_x - self.min_x)

    @property
    def height(self) -> float:
        return max(1.0, self.max_y - self.min_y)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)

    @property
    def orientation(self) -> str:
        if self.height >= self.width * 1.18:
            return "v"
        if self.width >= self.height * 1.18:
            return "h"
        metadata_orientation = self.metadata.get("orientation") or self.metadata.get("direction")
        if metadata_orientation in {"vertical", "v"}:
            return "v"
        if metadata_orientation in {"horizontal", "h"}:
            return "h"
        return "v" if self.height >= self.width else "h"

    @property
    def font_size(self) -> float:
        return max(1.0, min(self.width, self.height))

    @property
    def angle_degrees(self) -> float:
        rect = _min_area_rect(self.points)
        return rect.angle_degrees

    @property
    def normalized_reading_order(self) -> int:
        return self.reading_order if self.reading_order is not None else 999_999

    def intersects_region(self, bbox: list[float], width: int, height: int) -> bool:
        rx, ry, rw, rh = [float(value) for value in bbox]
        left = rx * width
        top = ry * height
        right = (rx + rw) * width
        bottom = (ry + rh) * height
        cx, cy = self.center
        return left <= cx <= right and top <= cy <= bottom


@dataclass(frozen=True)
class _MinRect:
    width: float
    height: float
    angle_degrees: float


def merge_textline_boxes(lines: list[TextLineBox], width: int, height: int) -> tuple[list[list[TextLineBox]], list[dict[str, Any]]]:
    if not lines:
        return [], []
    parent = list(range(len(lines)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edges_debug: list[dict[str, Any]] = []
    for (left_index, left), (right_index, right) in itertools.combinations(enumerate(lines), 2):
        decision = can_merge_textlines(left, right)
        if decision["can_merge"]:
            union(left_index, right_index)
        edges_debug.append(
            {
                "u": left.item_id,
                "v": right.item_id,
                "can_merge": decision["can_merge"],
                "reason": decision["reason"],
                "layout_partition_id": decision.get("layout_partition_id"),
            }
        )

    groups_by_root: dict[int, list[TextLineBox]] = {}
    for index, line in enumerate(lines):
        groups_by_root.setdefault(find(index), []).append(line)

    groups = [_sort_group(group) for group in groups_by_root.values()]
    groups.sort(key=lambda group: min(line.normalized_reading_order for line in group))
    return groups, edges_debug


def can_merge_textlines(left: TextLineBox, right: TextLineBox) -> dict[str, Any]:
    if _explicit_same_region(left, right):
        return {"can_merge": False, "reason": "explicit_region_preserves_textlines"}
    if _region_mismatch(left, right):
        return {"can_merge": False, "reason": "different_layout_or_bubble_region"}
    if left.orientation != right.orientation:
        return {"can_merge": False, "reason": "orientation_mismatch"}
    if _font_ratio(left, right) > 1.75:
        return {"can_merge": False, "reason": "font_size_ratio"}
    if _angle_delta(left.angle_degrees, right.angle_degrees) > 14:
        return {"can_merge": False, "reason": "angle_delta"}
    if _punctuationish(left.text) or _punctuationish(right.text):
        return {"can_merge": False, "reason": "punctuation_or_empty"}
    if left.orientation == "v":
        return _can_merge_vertical(left, right)
    return _can_merge_horizontal(left, right)


def polygon_from_bbox(bbox: list[float], width: int, height: int) -> np.ndarray:
    x, y, w, h = bbox
    return np.array(
        [
            [x * width, y * height],
            [(x + w) * width, y * height],
            [(x + w) * width, (y + h) * height],
            [x * width, (y + h) * height],
        ],
        dtype=np.float32,
    )


def polygon_to_normalized_bbox(points: Iterable[Iterable[float]], width: int, height: int) -> list[float]:
    pts = np.array(list(points), dtype=np.float32)
    min_coord = np.min(pts, axis=0)
    max_coord = np.max(pts, axis=0)
    x = _clip(min_coord[0] / width)
    y = _clip(min_coord[1] / height)
    right = _clip(max_coord[0] / width)
    bottom = _clip(max_coord[1] / height)
    return _round_bbox([x, y, max(0.0, right - x), max(0.0, bottom - y)])


def normalize_polygon(points: Iterable[Iterable[float]], width: int, height: int) -> list[list[float]]:
    return [[round(_clip(float(x) / width), 6), round(_clip(float(y) / height), 6)] for x, y in points]


def union_bbox(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return [0, 0, 0, 0]
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return _round_bbox([left, top, max(0.0, right - left), max(0.0, bottom - top)])


def _can_merge_vertical(left: TextLineBox, right: TextLineBox) -> dict[str, Any]:
    overlap = _interval_overlap(left.min_y, left.max_y, right.min_y, right.max_y) / max(1.0, min(left.height, right.height))
    horizontal_gap = max(0.0, max(left.min_x, right.min_x) - min(left.max_x, right.max_x))
    center_gap = abs(left.center[0] - right.center[0])
    baseline_gap = abs(left.center[1] - right.center[1])
    char_size = max(1.0, min(left.font_size, right.font_size))
    if overlap < 0.45:
        return {"can_merge": False, "reason": "vertical_y_overlap"}
    if baseline_gap > char_size * 0.55:
        return {"can_merge": False, "reason": "vertical_baseline_gap"}
    if horizontal_gap > char_size * 1.6 or center_gap > char_size * 2.6:
        return {"can_merge": False, "reason": "vertical_column_gap"}
    return {"can_merge": True, "reason": "vertical_columns_close"}


def _can_merge_horizontal(left: TextLineBox, right: TextLineBox) -> dict[str, Any]:
    overlap = _interval_overlap(left.min_x, left.max_x, right.min_x, right.max_x) / max(1.0, min(left.width, right.width))
    vertical_gap = max(0.0, max(left.min_y, right.min_y) - min(left.max_y, right.max_y))
    center_gap = abs(left.center[1] - right.center[1])
    char_size = max(1.0, min(left.font_size, right.font_size))
    if overlap < 0.35:
        return {"can_merge": False, "reason": "horizontal_x_overlap"}
    if vertical_gap > char_size * 1.6 or center_gap > char_size * 2.8:
        return {"can_merge": False, "reason": "horizontal_row_gap"}
    return {"can_merge": True, "reason": "horizontal_rows_close"}


def _sort_group(group: list[TextLineBox]) -> list[TextLineBox]:
    if not group:
        return group
    vertical_votes = sum(1 for line in group if line.orientation == "v")
    if vertical_votes >= len(group) / 2:
        return sorted(group, key=lambda line: (line.normalized_reading_order, -line.center[0], line.center[1]))
    return sorted(group, key=lambda line: (line.normalized_reading_order, line.center[1], line.center[0]))


def _explicit_same_region(left: TextLineBox, right: TextLineBox) -> bool:
    layout_left = left.metadata.get("layout_region_id")
    layout_right = right.metadata.get("layout_region_id")
    bubble_left = left.metadata.get("bubble_id")
    bubble_right = right.metadata.get("bubble_id")
    if layout_left and layout_left == layout_right:
        return True
    if bubble_left and bubble_left == bubble_right:
        return True
    return bool(left.source_region_id and left.source_region_id == right.source_region_id)


def _region_mismatch(left: TextLineBox, right: TextLineBox) -> bool:
    for key in ("layout_region_id", "bubble_id"):
        lval = left.metadata.get(key)
        rval = right.metadata.get(key)
        if lval and rval and lval != rval:
            return True
    return False


def _font_ratio(left: TextLineBox, right: TextLineBox) -> float:
    small = max(1.0, min(left.font_size, right.font_size))
    return max(left.font_size, right.font_size) / small


def _angle_delta(left: float, right: float) -> float:
    delta = abs(left - right) % 180.0
    return min(delta, 180.0 - delta)


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _punctuationish(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return not any("\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff" or char.isalnum() for char in stripped)


def _min_area_rect(points: np.ndarray) -> _MinRect:
    pts = np.asarray(points, dtype=np.float32)
    best: _MinRect | None = None
    for index in range(len(pts)):
        p0 = pts[index]
        p1 = pts[(index + 1) % len(pts)]
        angle = math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))
        cos_v = math.cos(-angle)
        sin_v = math.sin(-angle)
        rotation = np.array([[cos_v, -sin_v], [sin_v, cos_v]], dtype=np.float32)
        rotated = pts @ rotation.T
        width = float(np.max(rotated[:, 0]) - np.min(rotated[:, 0]))
        height = float(np.max(rotated[:, 1]) - np.min(rotated[:, 1]))
        area = width * height
        if best is None or area < best.width * best.height:
            best = _MinRect(max(width, 1.0), max(height, 1.0), math.degrees(angle))
    return best or _MinRect(1.0, 1.0, 0.0)


def _round_bbox(box: list[float]) -> list[float]:
    return [round(_clip(value), 6) for value in box]


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
