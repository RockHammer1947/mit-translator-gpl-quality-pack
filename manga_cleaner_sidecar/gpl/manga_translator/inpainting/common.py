from __future__ import annotations

from abc import ABC
from pathlib import Path


class OfflineInpainter(ABC):
    _MODEL_SUB_DIR = "inpainting"

    def __init__(self, model_dir: str | None = None, *args, **kwargs) -> None:
        super().__init__()
        self.model_dir = model_dir or "."

    def _get_file_path(self, filename: str) -> str:
        return str(Path(self.model_dir) / filename)

