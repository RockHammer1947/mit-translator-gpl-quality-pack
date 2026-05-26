from __future__ import annotations

from pathlib import Path

from comic_detector_sidecar.contracts.documents import DetectorOptions
from comic_detector_sidecar.providers.base import ComicDetectionResult, ComicDetectorProvider
from comic_detector_sidecar.providers.mit_ctd import MitCtdComicDetectorProvider


class CtdGplComicDetectorProvider(ComicDetectorProvider):
    name = "ctd-gpl"

    def __init__(self) -> None:
        self._delegate = MitCtdComicDetectorProvider()

    def is_available(self, options: DetectorOptions) -> bool:
        return self._delegate.is_available(options)

    def doctor(self, options: DetectorOptions) -> dict:
        payload = self._delegate.doctor(options)
        payload["name"] = self.name
        payload["delegate"] = "mit-ctd"
        payload["license"] = "GPL-3.0 quality-pack baseline"
        return payload

    def detect_image(self, image_path: Path, options: DetectorOptions) -> ComicDetectionResult:
        result = self._delegate.detect_image(image_path, options)
        for item in [*result.regions, *result.textlines]:
            item.provider = self.name
            item.metadata["delegate_provider"] = "mit-ctd"
        for item in result.bubbles or []:
            item.metadata["delegate_provider"] = "mit-ctd"
        for item in result.layout_regions or []:
            item.metadata["delegate_provider"] = "mit-ctd"
        if result.quality is None:
            result.quality = {}
        result.quality["provider"] = self.name
        result.quality["delegate_provider"] = "mit-ctd"
        result.quality["license"] = "GPL-3.0 quality-pack baseline"
        return result
