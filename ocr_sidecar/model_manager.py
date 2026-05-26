import hashlib
import os
import subprocess
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Optional

from ocr_sidecar.protocol import SidecarError, emit_event


MIT_48PX_LICENSE = "GPLv3 external adapter/runtime"
MIT_48PX_INTERNAL_LICENSE = "GPLv3 MIT-derived internal runtime"
MIT_MANGA_HYBRID_LICENSE = "GPLv3 MIT-derived runtime + manga-ocr fallback"
MIT_48PX_FILES = [
    {
        "name": "ocr_ar_48px.ckpt",
        "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr_ar_48px.ckpt",
        "sha256": "29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d",
        "size_mb": 127.0,
    },
    {
        "name": "alphabet-all-v7.txt",
        "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/alphabet-all-v7.txt",
        "sha256": "f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a",
        "size_mb": 0.1,
    },
]


EventSink = Callable[[dict[str, Any]], None]


@dataclass
class EngineModelStatus:
    name: str
    installed: bool
    model_ready: bool
    needs_download: bool
    available: bool
    reason: Optional[str] = None
    model_dir: Optional[Path] = None
    adapter_path: Optional[Path] = None
    download_size_mb: Optional[float] = None
    license: Optional[str] = None

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
        if self.model_dir:
            payload["model_dir"] = str(self.model_dir)
        if self.adapter_path:
            payload["adapter_path"] = str(self.adapter_path)
        if self.download_size_mb is not None:
            payload["download_size_mb"] = self.download_size_mb
        if self.license:
            payload["license"] = self.license
        return payload


def model_root() -> Path:
    override = os.environ.get("OCR_SIDECAR_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "com.taruiv2.platform" / "models" / "ocr"
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "com.taruiv2.platform" / "models" / "ocr"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "com.taruiv2.platform" / "models" / "ocr"


def engine_model_dir(engine: str) -> Path:
    return model_root() / engine


def resolve_mit_48px_adapter() -> Optional[Path]:
    override = os.environ.get("OCR_MIT_48PX_ADAPTER")
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.exists() and os.access(candidate, os.X_OK) else None
    found = shutil.which("mit-48px-ocr-adapter")
    return Path(found) if found else None


def resolve_mit_48px_internal_runtime() -> Optional[Path]:
    override = os.environ.get("OCR_MIT_REPO")
    if override:
        candidate = Path(override).expanduser()
        return candidate if _looks_like_mit_repo(candidate) else None
    candidates = [
        Path.cwd() / "对标项目" / "manga-image-translator-main",
        Path.cwd() / "manga-image-translator-main",
    ]
    for parent in Path(__file__).resolve().parents:
        candidates.extend(
            [
                parent / "对标项目" / "manga-image-translator-main",
                parent / "manga-image-translator-main",
            ]
        )
    for candidate in candidates:
        if _looks_like_mit_repo(candidate):
            return candidate
    return None


def status_for_engine(engine: str) -> EngineModelStatus:
    if engine in {"dummy-static", "dummy-empty", "dummy"}:
        return EngineModelStatus(
            name=engine,
            installed=True,
            model_ready=True,
            needs_download=False,
            available=True,
        )
    if engine == "mit-manga-hybrid":
        mit_status = status_for_engine("mit-48px-internal")
        manga_status = status_for_engine("manga-ocr")
        reason = None
        if not mit_status.available:
            reason = mit_status.reason
        elif not manga_status.available:
            reason = manga_status.reason
        return EngineModelStatus(
            name=engine,
            installed=mit_status.installed and manga_status.installed,
            model_ready=mit_status.model_ready and manga_status.model_ready,
            needs_download=mit_status.needs_download or manga_status.needs_download,
            available=mit_status.available and manga_status.available,
            reason=reason,
            model_dir=engine_model_dir("mit-48px"),
            adapter_path=mit_status.adapter_path,
            download_size_mb=mit_status.download_size_mb,
            license=MIT_MANGA_HYBRID_LICENSE,
        )
    if engine in {"mit-48px", "mit-48px-internal"}:
        if engine == "mit-48px-internal":
            runtime = resolve_mit_48px_internal_runtime()
            model_dir = engine_model_dir("mit-48px")
            missing = [spec["name"] for spec in MIT_48PX_FILES if not (model_dir / str(spec["name"])).exists()]
            model_ready = not missing
            missing_deps = _missing_mit_48px_internal_deps()
            reason = None
            if runtime is None:
                reason = "manga-image-translator runtime not found. Set OCR_MIT_REPO."
            elif missing_deps:
                reason = f"missing Python dependencies: {', '.join(missing_deps)}"
            elif not model_ready:
                reason = f"missing model files: {', '.join(missing)}"
            return EngineModelStatus(
                name=engine,
                installed=runtime is not None and not missing_deps,
                model_ready=model_ready,
                needs_download=not model_ready,
                available=runtime is not None and not missing_deps and model_ready,
                reason=reason,
                model_dir=model_dir,
                adapter_path=runtime,
                download_size_mb=sum(float(spec["size_mb"]) for spec in MIT_48PX_FILES) if not model_ready else 0.0,
                license=MIT_48PX_INTERNAL_LICENSE,
            )
        adapter = resolve_mit_48px_adapter()
        model_dir = engine_model_dir(engine)
        missing = [spec["name"] for spec in MIT_48PX_FILES if not (model_dir / str(spec["name"])).exists()]
        model_ready = not missing
        installed = adapter is not None
        reason = None
        if not installed:
            reason = "mit-48px adapter not found. Set OCR_MIT_48PX_ADAPTER or install mit-48px-ocr-adapter."
        elif not model_ready:
            reason = f"missing model files: {', '.join(missing)}"
        return EngineModelStatus(
            name=engine,
            installed=installed,
            model_ready=model_ready,
            needs_download=not model_ready,
            available=installed and model_ready,
            reason=reason,
            model_dir=model_dir,
            adapter_path=adapter,
            download_size_mb=sum(float(spec["size_mb"]) for spec in MIT_48PX_FILES) if not model_ready else 0.0,
            license=MIT_48PX_LICENSE,
        )
    if engine == "manga-ocr":
        installed = _can_import("manga_ocr")
        return EngineModelStatus(
            name=engine,
            installed=installed,
            model_ready=installed,
            needs_download=False,
            available=installed,
            reason=None if installed else "manga-ocr is not installed",
            model_dir=engine_model_dir(engine),
            license="Apache-2.0 package; model license inherited from Hugging Face model",
        )
    if engine in {"paddle", "manga-tiled"}:
        installed = _can_import("paddleocr")
        return EngineModelStatus(
            name=engine,
            installed=installed,
            model_ready=installed,
            needs_download=False,
            available=installed,
            reason=None if installed else "paddleocr is not installed",
        )
    return EngineModelStatus(
        name=engine,
        installed=False,
        model_ready=False,
        needs_download=False,
        available=False,
        reason=f"OCR engine '{engine}' is not supported.",
    )


def all_engine_statuses() -> list[EngineModelStatus]:
    return [
        status_for_engine("dummy-static"),
        status_for_engine("dummy-empty"),
        status_for_engine("dummy"),
        status_for_engine("mit-manga-hybrid"),
        status_for_engine("mit-48px-internal"),
        status_for_engine("mit-48px"),
        status_for_engine("manga-ocr"),
        status_for_engine("manga-tiled"),
        status_for_engine("paddle"),
    ]


def select_best_engine() -> str:
    for engine in ("mit-manga-hybrid", "mit-48px-internal", "mit-48px", "manga-ocr", "paddle", "manga-tiled"):
        if status_for_engine(engine).available:
            return engine
    raise SidecarError("OCR_ENGINE_LOAD_FAILED", "No real OCR engine is available.")


def prepare_engine_models(engine: str, job_id: str, jsonl: bool, download: bool = True) -> str:
    requested = engine
    if requested == "auto":
        requested = _select_prepare_target()

    emit_event(
        {
            "type": "started",
            "job_id": job_id,
            "stage": "ocr_prepare",
            "engine": requested,
        },
        jsonl,
    )

    if requested == "mit-manga-hybrid":
        _prepare_mit_48px(job_id, jsonl, download, engine_name="mit-48px-internal")
        _prepare_manga_ocr(job_id, jsonl, download)
        status = status_for_engine("mit-manga-hybrid")
    elif requested in {"mit-48px", "mit-48px-internal"}:
        status = _prepare_mit_48px(job_id, jsonl, download, engine_name=requested)
    elif requested == "manga-ocr":
        status = _prepare_manga_ocr(job_id, jsonl, download)
    else:
        status = status_for_engine(requested)

    emit_event({"type": "model_status", "job_id": job_id, **status.to_doctor_payload()}, jsonl)
    selected = status.name if status.available else (_fallback_after_prepare(status.name) if engine == "auto" else None)
    if selected is None and download:
        raise SidecarError("OCR_ENGINE_LOAD_FAILED", status.reason or f"OCR engine {status.name} is not ready.")
    emit_event(
        {
            "type": "done",
            "job_id": job_id,
            "status": "completed",
            "manifest_path": None,
            "selected_engine": selected,
            "requested_engine": engine,
        },
        jsonl,
    )
    return selected


def _select_prepare_target() -> str:
    hybrid = status_for_engine("mit-manga-hybrid")
    if hybrid.installed:
        return "mit-manga-hybrid"
    mit_internal = status_for_engine("mit-48px-internal")
    if mit_internal.installed:
        return "mit-48px-internal"
    mit = status_for_engine("mit-48px")
    if mit.installed:
        return "mit-48px"
    manga = status_for_engine("manga-ocr")
    if manga.installed:
        return "manga-ocr"
    return select_best_engine()


def _fallback_after_prepare(prepared_engine: str) -> str:
    status = status_for_engine(prepared_engine)
    if status.available:
        return prepared_engine
    return select_best_engine()


def _prepare_mit_48px(job_id: str, jsonl: bool, download: bool, engine_name: str = "mit-48px") -> EngineModelStatus:
    status = status_for_engine(engine_name)
    emit_event({"type": "model_status", "job_id": job_id, **status.to_doctor_payload()}, jsonl)
    if not status.needs_download:
        return status_for_engine(engine_name)
    if not download:
        return status
    model_dir = engine_model_dir("mit-48px")
    model_dir.mkdir(parents=True, exist_ok=True)
    total = len(MIT_48PX_FILES)
    for index, spec in enumerate(MIT_48PX_FILES, 1):
        destination = model_dir / str(spec["name"])
        if destination.exists() and _sha256(destination) == spec["sha256"]:
            continue
        emit_event(
            {
                "type": "progress",
                "job_id": job_id,
                "stage": "ocr_model_download",
                "progress": (index - 1) / total,
                "message": f"Downloading {spec['name']}",
            },
            jsonl,
        )
        _download_file(str(spec["url"]), destination, str(spec["sha256"]))
        emit_event({"type": "artifact", "job_id": job_id, "kind": "ocr_model_file", "path": str(destination)}, jsonl)
    emit_event(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "ocr_model_download",
            "progress": 1.0,
            "message": "OCR model files are ready",
        },
        jsonl,
    )
    return status_for_engine(engine_name)


def _prepare_manga_ocr(job_id: str, jsonl: bool, download: bool) -> EngineModelStatus:
    status = status_for_engine("manga-ocr")
    emit_event({"type": "model_status", "job_id": job_id, **status.to_doctor_payload()}, jsonl)
    if not status.installed:
        return status
    if not download:
        return status
    emit_event(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "ocr_model_warmup",
            "progress": 0.5,
            "message": "Preparing manga-ocr model cache",
        },
        jsonl,
    )
    try:
        _warmup_manga_ocr_subprocess()
    except Exception as e:
        return EngineModelStatus(
            name="manga-ocr",
            installed=True,
            model_ready=False,
            needs_download=True,
            available=False,
            reason=f"manga-ocr warmup failed: {e}",
            model_dir=engine_model_dir("manga-ocr"),
            license=status.license,
        )
    emit_event(
        {
            "type": "progress",
            "job_id": job_id,
            "stage": "ocr_model_warmup",
            "progress": 1.0,
            "message": "manga-ocr is ready",
        },
        jsonl,
    )
    return status_for_engine("manga-ocr")


def _warmup_manga_ocr_subprocess() -> None:
    timeout = int(os.environ.get("OCR_MANGA_OCR_WARMUP_TIMEOUT_SEC", "600"))
    code = """
import os
import sys
from contextlib import redirect_stdout
with redirect_stdout(sys.stderr):
    from manga_ocr import MangaOcr
    MangaOcr()
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
    )
    if completed.stdout:
        print(completed.stdout, file=sys.stderr, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"manga-ocr warmup exited with code {completed.returncode}")


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
            raise SidecarError(
                "OCR_ENGINE_LOAD_FAILED",
                f"Downloaded model hash mismatch for {destination.name}: {actual}",
            )
        partial.replace(destination)
    except (OSError, urllib.error.URLError) as e:
        partial.unlink(missing_ok=True)
        raise SidecarError("OCR_ENGINE_LOAD_FAILED", f"Failed to download OCR model {url}: {e}") from e


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _can_import(module_name: str) -> bool:
    if module_name == "paddleocr":
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        with redirect_stdout(sys.stderr):
            __import__(module_name)
        return True
    except Exception:
        return False


def _looks_like_mit_repo(path: Path) -> bool:
    return (path / "manga_translator" / "ocr" / "model_48px.py").exists()


def _missing_mit_48px_internal_deps() -> list[str]:
    missing = []
    for module_name in (
        "torch",
        "torchvision",
        "einops",
        "networkx",
        "py3langid",
        "shapely",
        "colorama",
        "dotenv",
        "langcodes",
        "omegaconf",
        "skimage",
        "timm",
    ):
        if not _can_import(module_name):
            missing.append(module_name)
    return missing
