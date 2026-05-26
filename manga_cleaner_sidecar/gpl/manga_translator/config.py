from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InpainterConfig:
    inpainting_precision: str = "bf16"

