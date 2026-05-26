import platform

from comic_detector_sidecar import __version__
from comic_detector_sidecar.contracts.documents import DetectorOptions, SCHEMA_VERSION
from comic_detector_sidecar.providers.apple_vision import AppleVisionComicDetectorProvider
from comic_detector_sidecar.providers.comic_text_detector import ComicTextDetectorProvider
from comic_detector_sidecar.providers.ctd_gpl import CtdGplComicDetectorProvider
from comic_detector_sidecar.providers.heuristic import HeuristicComicDetectorProvider
from comic_detector_sidecar.providers.mit_ctd import MitCtdComicDetectorProvider
from comic_detector_sidecar.providers.mmocr import MmocrComicDetectorProvider
from comic_detector_sidecar.providers.ogkalu_rtdetr import OgkaluRtdetrComicDetectorProvider
from comic_detector_sidecar.providers.paddle_layout import PaddleLayoutComicDetectorProvider


def run_doctor(options: DetectorOptions) -> dict:
    providers = [
        HeuristicComicDetectorProvider().doctor(options),
        CtdGplComicDetectorProvider().doctor(options),
        MitCtdComicDetectorProvider().doctor(options),
        ComicTextDetectorProvider().doctor(options),
        MmocrComicDetectorProvider().doctor(options),
        OgkaluRtdetrComicDetectorProvider().doctor(options),
        PaddleLayoutComicDetectorProvider().doctor(options),
        AppleVisionComicDetectorProvider().doctor(options),
    ]
    return {
        "type": "doctor",
        "job_id": "doctor",
        "schema_version": SCHEMA_VERSION,
        "sidecar_version": __version__,
        "python_version": platform.python_version(),
        "providers": providers,
    }
