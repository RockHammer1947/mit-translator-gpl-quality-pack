from comic_detector_sidecar.providers.base import ComicDetectorProvider
from comic_detector_sidecar.providers.apple_vision import AppleVisionComicDetectorProvider
from comic_detector_sidecar.providers.comic_text_detector import ComicTextDetectorProvider
from comic_detector_sidecar.providers.ctd_gpl import CtdGplComicDetectorProvider
from comic_detector_sidecar.providers.heuristic import HeuristicComicDetectorProvider
from comic_detector_sidecar.providers.mit_ctd import MitCtdComicDetectorProvider
from comic_detector_sidecar.providers.mmocr import MmocrComicDetectorProvider
from comic_detector_sidecar.providers.ogkalu_rtdetr import OgkaluRtdetrComicDetectorProvider
from comic_detector_sidecar.providers.paddle_layout import PaddleLayoutComicDetectorProvider


def get_provider(name: str) -> ComicDetectorProvider:
    if name == "heuristic":
        return HeuristicComicDetectorProvider()
    if name == "ctd-gpl":
        return CtdGplComicDetectorProvider()
    if name == "comic-text-detector":
        return ComicTextDetectorProvider()
    if name == "mit-ctd":
        return MitCtdComicDetectorProvider()
    if name == "mmocr":
        return MmocrComicDetectorProvider()
    if name == "ogkalu-rtdetr":
        return OgkaluRtdetrComicDetectorProvider()
    if name == "paddle-layout":
        return PaddleLayoutComicDetectorProvider()
    if name == "apple-vision":
        return AppleVisionComicDetectorProvider()
    raise ValueError(f"Comic detector provider '{name}' is not supported")
