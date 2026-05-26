from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from comic_detector_sidecar.contracts.documents import BubbleRegion, ComicTextRegion, DetectorOptions, LayoutRegion


@dataclass
class ComicDetectionResult:
    regions: list[ComicTextRegion]
    textlines: list[ComicTextRegion]
    raw_mask: np.ndarray | None = None
    refined_mask: np.ndarray | None = None
    bubble_mask: np.ndarray | None = None
    bubbles: list[BubbleRegion] | None = None
    layout_regions: list[LayoutRegion] | None = None
    quality: dict | None = None


class ComicDetectorProvider(ABC):
    name: str

    @abstractmethod
    def is_available(self, options: DetectorOptions) -> bool:
        pass

    @abstractmethod
    def doctor(self, options: DetectorOptions) -> dict:
        pass

    @abstractmethod
    def detect_image(self, image_path: Path, options: DetectorOptions) -> ComicDetectionResult:
        pass
