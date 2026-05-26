from __future__ import annotations

import hashlib
import os
import contextlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comic_detector_sidecar.utils.jsonl import emit_event

DEFAULT_CTD_MODEL_NAME = "comictextdetector.pt.onnx"
DEFAULT_CTD_MODEL_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/"
    "comictextdetector.pt.onnx"
)
DEFAULT_CTD_MODEL_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"
DEFAULT_CTD_MODEL_SIZE_MB = 90.3

DEFAULT_OGKALU_RTDETR_MODEL_NAME = "detector-v4-s_int8.onnx"
DEFAULT_OGKALU_RTDETR_MODEL_URL = (
    "https://huggingface.co/ogkalu/comic-text-and-bubble-detector/resolve/main/"
    "detector-v4-s_int8.onnx"
)
DEFAULT_OGKALU_RTDETR_MODEL_SHA256 = "5fe9e4f576e49d4e7e8b0e029d6d3cdc252abd4694113e1cae120e62c931ea79"
DEFAULT_OGKALU_RTDETR_MODEL_SIZE_MB = 10.7


@dataclass
class ComicDetectorModelStatus:
    name: str
    installed: bool
    model_ready: bool
    needs_download: bool
    available: bool
    reason: str | None = None
    model_path: Path | None = None
    adapter_command: str | None = None
    download_size_mb: float | None = None
    license: str | None = None

    def to_doctor_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "available": self.available,
            "installed": self.installed,
            "model_ready": self.model_ready,
            "needs_download": self.needs_download,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.model_path:
            payload["model_path"] = str(self.model_path)
        if self.adapter_command:
            payload["adapter_command"] = self.adapter_command
        if self.download_size_mb is not None:
            payload["download_size_mb"] = self.download_size_mb
        if self.license:
            payload["license"] = self.license
        return payload


def model_root() -> Path:
    override = os.environ.get("COMIC_DETECTOR_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "com.taruiv2.platform" / "models" / "comic-detector"
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "com.taruiv2.platform" / "models" / "comic-detector"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "com.taruiv2.platform" / "models" / "comic-detector"


def default_comic_text_detector_model_path() -> Path:
    return model_root() / "comic-text-detector" / DEFAULT_CTD_MODEL_NAME


def default_ogkalu_rtdetr_model_path() -> Path:
    return model_root() / "ogkalu-rtdetr" / DEFAULT_OGKALU_RTDETR_MODEL_NAME


def bundled_adapter_command() -> str:
    return f"{sys.executable} -m comic_detector_sidecar.adapters.comic_text_detector_adapter"


def status_for_comic_text_detector(model_path: Path | None = None, adapter_command: str | None = None) -> ComicDetectorModelStatus:
    resolved_model_path = (model_path.expanduser() if model_path else default_comic_text_detector_model_path())
    resolved_adapter = adapter_command or os.environ.get("COMIC_TEXT_DETECTOR_CMD") or bundled_adapter_command()
    model_ready = resolved_model_path.exists()
    reason = None if model_ready else f"model not found: {resolved_model_path}"
    return ComicDetectorModelStatus(
        name="comic-text-detector",
        installed=True,
        model_ready=model_ready,
        needs_download=not model_ready,
        available=model_ready,
        reason=reason,
        model_path=resolved_model_path,
        adapter_command=resolved_adapter,
        download_size_mb=0.0 if model_ready else DEFAULT_CTD_MODEL_SIZE_MB,
        license="GPLv3 upstream model/runtime; clean-room ONNX adapter",
    )


def status_for_ogkalu_rtdetr(model_path: Path | None = None) -> ComicDetectorModelStatus:
    resolved_model_path = (model_path.expanduser() if model_path else default_ogkalu_rtdetr_model_path())
    model_ready = resolved_model_path.exists()
    runtime_ready = _can_import("onnxruntime")
    reason = None
    if not runtime_ready:
        reason = "onnxruntime is not installed"
    elif not model_ready:
        reason = f"model not found: {resolved_model_path}"
    return ComicDetectorModelStatus(
        name="ogkalu-rtdetr",
        installed=runtime_ready,
        model_ready=model_ready,
        needs_download=not model_ready,
        available=runtime_ready and model_ready,
        reason=reason,
        model_path=resolved_model_path,
        adapter_command=None,
        download_size_mb=0.0 if model_ready else DEFAULT_OGKALU_RTDETR_MODEL_SIZE_MB,
        license="Apache-2.0 model card; verify upstream training data before commercial default",
    )


def status_for_paddle_layout() -> ComicDetectorModelStatus:
    runtime_ready = _can_import("paddlex") and _can_import("paddle")
    model_path = Path.home() / ".paddlex" / "official_models" / "PP-DocLayoutV3"
    model_ready = model_path.exists()
    reason = None
    if not runtime_ready:
        reason = "paddlex/paddlepaddle is not installed"
    elif not model_ready:
        reason = f"model not found: {model_path}"
    return ComicDetectorModelStatus(
        name="paddle-layout",
        installed=runtime_ready,
        model_ready=model_ready,
        needs_download=runtime_ready and not model_ready,
        available=runtime_ready and model_ready,
        reason=reason,
        model_path=model_path,
        download_size_mb=126.0 if not model_ready else 0.0,
        license="Apache-2.0 upstream PaddleOCR/PaddleX",
    )


def status_for_mmocr() -> ComicDetectorModelStatus:
    runtime_ready, reason = _mmocr_runtime_ready()
    return ComicDetectorModelStatus(
        name="mmocr",
        installed=runtime_ready,
        model_ready=runtime_ready,
        needs_download=False,
        available=runtime_ready,
        reason=reason,
        model_path=Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        download_size_mb=None,
        license="Apache-2.0 upstream OpenMMLab code; model zoo weights follow upstream model metadata",
    )


def prepare_comic_text_detector(job_id: str, jsonl: bool, download: bool = True) -> ComicDetectorModelStatus:
    status = status_for_comic_text_detector()
    emit_event({"type": "model_status", "job_id": job_id, **status.to_doctor_payload()}, jsonl)
    if not status.needs_download:
        return status
    if not download:
        return status

    destination = default_comic_text_detector_model_path()
    emit_event(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "comic_detector_model_download",
            "progress": 0.0,
            "message": f"Downloading {DEFAULT_CTD_MODEL_NAME}",
        },
        jsonl,
    )
    _download_file(DEFAULT_CTD_MODEL_URL, destination, DEFAULT_CTD_MODEL_SHA256)
    emit_event({"type": "artifact", "job_id": job_id, "kind": "comic_detector_model", "path": str(destination)}, jsonl)
    emit_event(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "comic_detector_model_download",
            "progress": 1.0,
            "message": "comic-text-detector model is ready",
        },
        jsonl,
    )
    return status_for_comic_text_detector()


def prepare_ogkalu_rtdetr(job_id: str, jsonl: bool, download: bool = True) -> ComicDetectorModelStatus:
    status = status_for_ogkalu_rtdetr()
    emit_event({"type": "model_status", "job_id": job_id, **status.to_doctor_payload()}, jsonl)
    if not status.needs_download:
        return status
    if not download:
        return status
    destination = default_ogkalu_rtdetr_model_path()
    emit_event(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "ogkalu_rtdetr_model_download",
            "progress": 0.0,
            "message": f"Downloading {DEFAULT_OGKALU_RTDETR_MODEL_NAME}",
        },
        jsonl,
    )
    _download_file(DEFAULT_OGKALU_RTDETR_MODEL_URL, destination, DEFAULT_OGKALU_RTDETR_MODEL_SHA256)
    emit_event({"type": "artifact", "job_id": job_id, "kind": "ogkalu_rtdetr_model", "path": str(destination)}, jsonl)
    emit_event(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "ogkalu_rtdetr_model_download",
            "progress": 1.0,
            "message": "ogkalu RT-DETR model is ready",
        },
        jsonl,
    )
    return status_for_ogkalu_rtdetr()


def prepare_paddle_layout(job_id: str, jsonl: bool, download: bool = True) -> ComicDetectorModelStatus:
    status = status_for_paddle_layout()
    emit_event({"type": "model_status", "job_id": job_id, **status.to_doctor_payload()}, jsonl)
    if not status.needs_download:
        return status
    if not download:
        return status
    emit_event(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "paddle_layout_model_download",
            "progress": 0.0,
            "message": "Preparing PP-DocLayoutV3",
        },
        jsonl,
    )
    try:
        import paddlex

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        with contextlib.redirect_stdout(sys.stderr):
            paddlex.create_model("PP-DocLayoutV3")
    except Exception as error:
        raise RuntimeError(f"Failed to prepare PP-DocLayoutV3: {error}") from error
    emit_event(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "paddle_layout_model_download",
            "progress": 1.0,
            "message": "PP-DocLayoutV3 is ready",
        },
        jsonl,
    )
    return status_for_paddle_layout()


def prepare_mmocr(job_id: str, jsonl: bool, download: bool = True) -> ComicDetectorModelStatus:
    status = status_for_mmocr()
    emit_event({"type": "model_status", "job_id": job_id, **status.to_doctor_payload()}, jsonl)
    if not download or not status.available:
        return status
    emit_event(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "mmocr_model_prepare",
            "progress": 0.0,
            "message": "Preparing MMOCR default text detection model",
        },
        jsonl,
    )
    try:
        from comic_detector_sidecar.adapters.mmocr_adapter import DEFAULT_MMOCR_TEXTDET_MODEL, _assert_full_mmcv_ops

        _assert_full_mmcv_ops()
        from mmocr.apis import TextDetInferencer

        with contextlib.redirect_stdout(sys.stderr):
            TextDetInferencer(model=DEFAULT_MMOCR_TEXTDET_MODEL, device="cpu")
    except Exception as error:
        raise RuntimeError(f"Failed to prepare MMOCR default model: {error}") from error
    emit_event(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "mmocr_model_prepare",
            "progress": 1.0,
            "message": "MMOCR default text detection model is ready",
        },
        jsonl,
    )
    return status_for_mmocr()


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _mmocr_runtime_ready() -> tuple[bool, str | None]:
    for module_name in ["torch", "mmengine", "mmcv", "mmdet", "mmocr"]:
        if not _can_import(module_name):
            return False, f"{module_name} is not installed"
    try:
        import mmcv._ext  # noqa: F401
    except Exception:
        return False, "full mmcv native ops are not installed; mmcv-lite is not enough for MMOCR"
    return True, None


def _download_file(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        actual = _sha256(partial)
        if actual.lower() != expected_sha256.lower():
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded comic detector model hash mismatch: {actual}")
        partial.replace(destination)
    except (OSError, urllib.error.URLError) as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download comic detector model: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
