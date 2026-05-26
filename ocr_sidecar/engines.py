import logging
import json
import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Optional

import numpy as np
import cv2

from ocr_sidecar.contracts import BBox, OcrTextBlock, Polygon
from ocr_sidecar.image_ops import crop_bbox, crop_for_ocr, load_image_cv2
from ocr_sidecar.model_manager import engine_model_dir, resolve_mit_48px_adapter, status_for_engine

logger = logging.getLogger("ocr_sidecar.engines")


class OcrEngineError(Exception):
    pass


class OcrEngine(ABC):
    name: str
    last_attempts: list[dict[str, Any]] = []
    last_selected_attempt_id: Optional[str] = None

    def warmup(self) -> None:
        pass

    @abstractmethod
    def recognize_image(
        self,
        image_path: Path,
        lang_hint: str = "auto",
        region: Optional[BBox] = None,
        polygon: Optional[Polygon] = None,
        crop_policy: Optional[str] = None,
        target_text_height: Optional[int] = None,
        padding_ratio: Optional[float] = None,
        preprocess: Optional[str] = None,
        probability_threshold: Optional[float] = None,
        debug_output_dir: Optional[Path] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[OcrTextBlock]:
        pass


class DummyStaticOcrEngine(OcrEngine):
    name = "dummy-static"

    def recognize_image(
        self,
        image_path: Path,
        lang_hint: str = "auto",
        region: Optional[BBox] = None,
        polygon: Optional[Polygon] = None,
        crop_policy: Optional[str] = None,
        target_text_height: Optional[int] = None,
        padding_ratio: Optional[float] = None,
        preprocess: Optional[str] = None,
        probability_threshold: Optional[float] = None,
        debug_output_dir: Optional[Path] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[OcrTextBlock]:
        bbox = region if region is not None else BBox([0.05, 0.75, 0.90, 0.18])
        return [
            OcrTextBlock(
                text="Hello Subtitle",
                confidence=0.95,
                bbox=bbox,
                engine=self.name,
                polygon=polygon,
                textline_polygons=[polygon] if polygon is not None else [],
                metadata=metadata or {},
            )
        ]


class DummyEmptyOcrEngine(OcrEngine):
    name = "dummy-empty"

    def recognize_image(
        self,
        image_path: Path,
        lang_hint: str = "auto",
        region: Optional[BBox] = None,
        polygon: Optional[Polygon] = None,
        crop_policy: Optional[str] = None,
        target_text_height: Optional[int] = None,
        padding_ratio: Optional[float] = None,
        preprocess: Optional[str] = None,
        probability_threshold: Optional[float] = None,
        debug_output_dir: Optional[Path] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[OcrTextBlock]:
        return []


class DummyOcrEngine(DummyStaticOcrEngine):
    name = "dummy"


class PaddleOcrEngine(OcrEngine):
    name = "paddle"

    def __init__(self) -> None:
        self._ocr = None
        self._current_lang = None

    def map_language(self, lang: str) -> str:
        lang = lang.lower().strip()
        if lang in ("zh", "zh-cn", "chs", "ch"):
            return "ch"
        if lang in ("zh-tw", "cht", "chinese_cht"):
            return "chinese_cht"
        if lang in ("ja", "jp", "japan", "japanese"):
            return "japan"
        if lang in ("ko", "kr", "korean"):
            return "korean"
        if lang in ("en", "english"):
            return "en"
        return "ch"

    def _get_ocr_instance(self, lang_code: str) -> Any:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            with redirect_stdout(sys.stderr):
                from paddleocr import PaddleOCR
        except ImportError as e:
            raise OcrEngineError(
                "PaddleOCR is not installed. Run `uv sync --extra paddle` in sidecars/ocr-sidecar."
            ) from e

        if self._ocr is None or self._current_lang != lang_code:
            init_attempts = [
                {"use_angle_cls": False, "lang": lang_code, "show_log": False},
                {"use_angle_cls": False, "lang": lang_code},
            ]
            last_error: Exception | None = None
            for kwargs in init_attempts:
                try:
                    with redirect_stdout(sys.stderr):
                        self._ocr = PaddleOCR(**kwargs)
                    self._current_lang = lang_code
                    return self._ocr
                except (TypeError, ValueError) as e:
                    last_error = e
                    logger.warning("PaddleOCR init failed with args %s: %s", kwargs, e)
                    continue
                except Exception as e:
                    raise OcrEngineError(f"Failed to initialize PaddleOCR: {e}") from e
            raise OcrEngineError(f"Failed to initialize PaddleOCR for lang={lang_code}: {last_error}")

        return self._ocr

    def recognize_image(
        self,
        image_path: Path,
        lang_hint: str = "auto",
        region: Optional[BBox] = None,
        polygon: Optional[Polygon] = None,
        crop_policy: Optional[str] = None,
        target_text_height: Optional[int] = None,
        padding_ratio: Optional[float] = None,
        preprocess: Optional[str] = None,
        probability_threshold: Optional[float] = None,
        debug_output_dir: Optional[Path] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[OcrTextBlock]:
        lang_code = self.map_language(lang_hint)
        ocr_inst = self._get_ocr_instance(lang_code)

        image = load_image_cv2(image_path)
        full_h, full_w = image.shape[:2]

        if region is not None:
            crop = crop_bbox(image, region.root)
            crop_h, crop_w = crop.shape[:2]
        else:
            crop = image
            crop_h, crop_w = full_h, full_w

        if crop is None or crop.size == 0:
            return []

        try:
            with redirect_stdout(sys.stderr):
                try:
                    results = ocr_inst.ocr(crop, cls=False)
                except TypeError:
                    results = ocr_inst.ocr(crop)
        except Exception as e:
            raise OcrEngineError(f"OCR execution failed: {e}") from e

        entries = _parse_paddle_results(results)
        blocks: list[OcrTextBlock] = []
        for bbox_pixels, text, confidence in entries:
            block = _to_text_block(
                bbox_pixels=bbox_pixels,
                text=text,
                confidence=confidence,
                crop_width=crop_w,
                crop_height=crop_h,
                region=region,
                engine_name=self.name,
            )
            if block is not None:
                blocks.append(block)
        return blocks


class MangaTiledOcrEngine(PaddleOcrEngine):
    name = "manga-tiled"

    def recognize_image(
        self,
        image_path: Path,
        lang_hint: str = "auto",
        region: Optional[BBox] = None,
        polygon: Optional[Polygon] = None,
        crop_policy: Optional[str] = None,
        target_text_height: Optional[int] = None,
        padding_ratio: Optional[float] = None,
        preprocess: Optional[str] = None,
        probability_threshold: Optional[float] = None,
        debug_output_dir: Optional[Path] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[OcrTextBlock]:
        if region is not None:
            return super().recognize_image(image_path=image_path, lang_hint=lang_hint, region=region)

        image = load_image_cv2(image_path)
        height, width = image.shape[:2]
        tile_regions = _manga_tile_regions(width=width, height=height)
        blocks: list[OcrTextBlock] = []
        for tile in tile_regions:
            blocks.extend(super().recognize_image(image_path=image_path, lang_hint=lang_hint, region=BBox(tile)))
        return _dedupe_text_blocks(blocks)


class MangaOcrEngine(OcrEngine):
    name = "manga-ocr"

    def __init__(self) -> None:
        self._ocr = None

    def _get_ocr_instance(self) -> Any:
        try:
            with redirect_stdout(sys.stderr):
                from manga_ocr import MangaOcr
        except ImportError as e:
            raise OcrEngineError(
                "manga-ocr is not installed. Run `uv sync --extra manga` in sidecars/ocr-sidecar."
            ) from e
        if self._ocr is None:
            try:
                with redirect_stdout(sys.stderr):
                    self._ocr = MangaOcr()
            except Exception as e:
                raise OcrEngineError(f"Failed to initialize manga-ocr: {e}") from e
        return self._ocr

    def recognize_image(
        self,
        image_path: Path,
        lang_hint: str = "auto",
        region: Optional[BBox] = None,
        polygon: Optional[Polygon] = None,
        crop_policy: Optional[str] = None,
        target_text_height: Optional[int] = None,
        padding_ratio: Optional[float] = None,
        preprocess: Optional[str] = None,
        probability_threshold: Optional[float] = None,
        debug_output_dir: Optional[Path] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[OcrTextBlock]:
        ocr_inst = self._get_ocr_instance()
        image = load_image_cv2(image_path)
        source_polygon = polygon.root if polygon is not None else None
        candidates: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(
            _manga_ocr_candidate_settings(
                crop_policy=crop_policy,
                target_text_height=target_text_height,
                padding_ratio=padding_ratio,
                preprocess=preprocess,
            ),
            1,
        ):
            crop, source_bbox, effective_polygon, crop_metadata = crop_for_ocr(
                image,
                region=region.root if region is not None else None,
                polygon=source_polygon,
                crop_policy=candidate["crop_policy"],
                orientation=_metadata_orientation(metadata),
                target_text_height=candidate["target_text_height"],
                padding_ratio=candidate["padding_ratio"],
                preprocess=candidate["preprocess"],
            )
            if crop is None or crop.size == 0:
                continue
            debug_metadata = {**(metadata or {}), "candidate_index": candidate_index}
            debug_crop_path = _write_debug_crop(debug_output_dir, debug_metadata, crop)
            crop_ink_ratio = _crop_ink_ratio(crop)
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                    cv2_ok = _write_crop(tmp.name, crop)
                    if not cv2_ok:
                        raise OcrEngineError("Failed to write temporary manga-ocr crop")
                    with redirect_stdout(sys.stderr):
                        text = str(ocr_inst(tmp.name)).strip()
            except OcrEngineError:
                raise
            except Exception as e:
                raise OcrEngineError(f"manga-ocr execution failed: {e}") from e
            candidate_metadata = {
                **crop_metadata,
                "requested_crop_policy": crop_policy,
                "candidate_crop_policy": candidate["crop_policy"],
                "candidate_index": candidate_index,
                "crop_ink_ratio": crop_ink_ratio,
            }
            _write_debug_crop_metadata(debug_crop_path, candidate_metadata, text)
            candidates.append(
                {
                    "text": text,
                    "score": _ocr_candidate_score(text),
                    "confidence": 0.86 if text else 0.0,
                    "source_bbox": source_bbox,
                    "effective_polygon": effective_polygon,
                    "metadata": candidate_metadata,
                    "debug_crop_path": debug_crop_path,
                }
            )
        if not candidates:
            self.last_attempts = []
            self.last_selected_attempt_id = None
            return []
        best = sorted(candidates, key=lambda item: item["score"], reverse=True)[0]
        text = str(best["text"]).strip()
        self.last_attempts = [
            {
                "attempt_id": f"manga_ocr_{candidate['metadata']['candidate_index']}",
                "recognizer": "manga-ocr",
                "status": "recognized" if str(candidate["text"]).strip() else "empty",
                "texts": [candidate["text"]] if str(candidate["text"]).strip() else [],
                "max_confidence": candidate["confidence"],
                "text_quality_score": candidate["score"],
                "crop_ink_ratio": candidate["metadata"].get("crop_ink_ratio"),
                **({"debug_crop_path": str(candidate["debug_crop_path"])} if candidate["debug_crop_path"] else {}),
            }
            for candidate in candidates
        ]
        self.last_selected_attempt_id = f"manga_ocr_{best['metadata']['candidate_index']}"
        if not text:
            return []
        bbox = BBox(best["source_bbox"])
        effective_polygon = best["effective_polygon"]
        crop_metadata = best["metadata"]
        debug_crop_path = best["debug_crop_path"]
        return [
            OcrTextBlock(
                text=text,
                confidence=0.86,
                bbox=bbox,
                engine=self.name,
                polygon=Polygon(effective_polygon) if effective_polygon is not None else None,
                textline_polygons=[Polygon(effective_polygon)] if effective_polygon is not None else [],
                metadata={
                    **(metadata or {}),
                    **crop_metadata,
                    "provider": "manga-ocr",
                    "confidence_source": "manga_ocr_no_native_confidence",
                    "attempt_id": f"manga_ocr_{crop_metadata['candidate_index']}",
                    "recognizer": "manga-ocr",
                    "text_quality_score": best["score"],
                    "crop_ink_ratio": crop_metadata.get("crop_ink_ratio"),
                    "candidate_count": len(candidates),
                    "candidate_scores": [
                        {"text": candidate["text"], "score": candidate["score"], "candidate_index": candidate["metadata"]["candidate_index"]}
                        for candidate in candidates
                    ],
                    **({"debug_crop_path": str(debug_crop_path)} if debug_crop_path else {}),
                },
            )
        ]


class Mit48pxAdapterOcrEngine(OcrEngine):
    name = "mit-48px"

    def recognize_image(
        self,
        image_path: Path,
        lang_hint: str = "auto",
        region: Optional[BBox] = None,
        polygon: Optional[Polygon] = None,
        crop_policy: Optional[str] = None,
        target_text_height: Optional[int] = None,
        padding_ratio: Optional[float] = None,
        preprocess: Optional[str] = None,
        probability_threshold: Optional[float] = None,
        debug_output_dir: Optional[Path] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[OcrTextBlock]:
        adapter = resolve_mit_48px_adapter()
        if adapter is None:
            raise OcrEngineError("mit-48px adapter not found. Set OCR_MIT_48PX_ADAPTER or install mit-48px-ocr-adapter.")
        status = status_for_engine(self.name)
        if not status.model_ready:
            raise OcrEngineError("mit-48px model files are missing. Run `ocr-sidecar prepare-models --engine mit-48px`.")

        payload = {
            "schema_version": "mit_48px_adapter_request.v1",
            "image_path": str(image_path),
            "language_hint": lang_hint,
            "region": region.root if region is not None else None,
            "polygon": polygon.root if polygon is not None else None,
            "crop_policy": crop_policy,
            "target_text_height": target_text_height,
            "padding_ratio": padding_ratio,
            "preprocess": preprocess,
            "probability_threshold": probability_threshold,
            "model_dir": str(engine_model_dir(self.name)),
            "debug_output_dir": str(debug_output_dir) if debug_output_dir else None,
            "metadata": metadata or {},
        }
        try:
            completed = subprocess.run(
                [str(adapter), "recognize-image"],
                input=json.dumps(payload, ensure_ascii=False),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=180,
            )
        except subprocess.TimeoutExpired as e:
            raise OcrEngineError("mit-48px adapter timed out") from e
        except OSError as e:
            raise OcrEngineError(f"Failed to run mit-48px adapter: {e}") from e
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if completed.returncode != 0:
            raise OcrEngineError(f"mit-48px adapter exited with code {completed.returncode}")
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as e:
            raise OcrEngineError("mit-48px adapter returned invalid JSON") from e
        raw_blocks = response.get("blocks", [])
        if not isinstance(raw_blocks, list):
            raise OcrEngineError("mit-48px adapter response must contain blocks[]")
        return [_block_from_adapter_payload(block, metadata or {}, engine_name=self.name) for block in raw_blocks]


class Mit48pxInternalOcrEngine(OcrEngine):
    name = "mit-48px-internal"

    def recognize_image(
        self,
        image_path: Path,
        lang_hint: str = "auto",
        region: Optional[BBox] = None,
        polygon: Optional[Polygon] = None,
        crop_policy: Optional[str] = None,
        target_text_height: Optional[int] = None,
        padding_ratio: Optional[float] = None,
        preprocess: Optional[str] = None,
        probability_threshold: Optional[float] = None,
        debug_output_dir: Optional[Path] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[OcrTextBlock]:
        status = status_for_engine(self.name)
        if not status.available:
            raise OcrEngineError(status.reason or "mit-48px-internal is not available")
        payload = {
            "schema_version": "mit_48px_internal_request.v1",
            "image_path": str(image_path),
            "language_hint": lang_hint,
            "region": region.root if region is not None else None,
            "polygon": polygon.root if polygon is not None else None,
            "crop_policy": crop_policy or "mit_textline",
            "target_text_height": target_text_height or 48,
            "padding_ratio": padding_ratio,
            "preprocess": preprocess,
            "probability_threshold": probability_threshold,
            **(
                {"probability_thresholds": metadata.get("mit_probability_thresholds")}
                if isinstance(metadata, dict) and isinstance(metadata.get("mit_probability_thresholds"), list)
                else {}
            ),
            "debug_output_dir": str(debug_output_dir) if debug_output_dir else None,
            "metadata": metadata or {},
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "ocr_sidecar.mit_48px_internal"],
                input=json.dumps(payload, ensure_ascii=False),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=180,
            )
        except subprocess.TimeoutExpired as e:
            raise OcrEngineError("mit-48px-internal timed out") from e
        except OSError as e:
            raise OcrEngineError(f"Failed to run mit-48px-internal: {e}") from e
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if completed.returncode != 0:
            raise OcrEngineError(f"mit-48px-internal exited with code {completed.returncode}")
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as e:
            raise OcrEngineError("mit-48px-internal returned invalid JSON") from e
        raw_blocks = response.get("blocks", [])
        if not isinstance(raw_blocks, list):
            raise OcrEngineError("mit-48px-internal response must contain blocks[]")
        self.last_attempts = response.get("attempts", []) if isinstance(response.get("attempts"), list) else []
        self.last_selected_attempt_id = response.get("selected_attempt_id") if isinstance(response.get("selected_attempt_id"), str) else None
        return [_block_from_adapter_payload(block, metadata or {}, engine_name=self.name) for block in raw_blocks]


class MitMangaHybridOcrEngine(OcrEngine):
    name = "mit-manga-hybrid"

    def __init__(self) -> None:
        self._mit = Mit48pxInternalOcrEngine()
        self._manga = MangaOcrEngine()

    def recognize_image(
        self,
        image_path: Path,
        lang_hint: str = "auto",
        region: Optional[BBox] = None,
        polygon: Optional[Polygon] = None,
        crop_policy: Optional[str] = None,
        target_text_height: Optional[int] = None,
        padding_ratio: Optional[float] = None,
        preprocess: Optional[str] = None,
        probability_threshold: Optional[float] = None,
        debug_output_dir: Optional[Path] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[OcrTextBlock]:
        status = status_for_engine(self.name)
        if not status.available:
            raise OcrEngineError(status.reason or "mit-manga-hybrid is not available")
        base_metadata = dict(metadata or {})
        retry_profile = str(base_metadata.get("retry_profile") or "mit_quality")
        if retry_profile == "mit_quality":
            thresholds = [probability_threshold, 0.05, 0.0]
        else:
            thresholds = [probability_threshold]
        thresholds = _dedupe_thresholds(thresholds)
        mit_blocks = self._mit.recognize_image(
            image_path=image_path,
            lang_hint=lang_hint,
            region=region,
            polygon=polygon,
            crop_policy=crop_policy,
            target_text_height=target_text_height,
            padding_ratio=padding_ratio,
            preprocess=preprocess,
            probability_threshold=probability_threshold,
            debug_output_dir=debug_output_dir,
            metadata={
                **base_metadata,
                "retry_profile": retry_profile,
                "mit_probability_thresholds": thresholds,
            },
        )
        attempts = list(self._mit.last_attempts or [])
        candidates: list[OcrTextBlock] = []
        for block in mit_blocks:
            block.metadata = {
                **block.metadata,
                "provider": self.name,
                "recognizer": "mit-48px-internal",
                "text_quality_score": _hybrid_text_quality_score(block.text, block.confidence),
            }
            candidates.append(block)

        manga_attempt_id = "manga_ocr_1"
        try:
            manga_blocks = self._manga.recognize_image(
                image_path=image_path,
                lang_hint=lang_hint,
                region=region,
                polygon=polygon,
                crop_policy=crop_policy or "mit_textline",
                target_text_height=target_text_height,
                padding_ratio=padding_ratio,
                preprocess=preprocess,
                debug_output_dir=debug_output_dir,
                metadata={**base_metadata, "attempt_id": manga_attempt_id, "recognizer": "manga-ocr"},
            )
            manga_attempts = list(self._manga.last_attempts or [])
            attempts.extend(
                manga_attempts
                or [
                    {
                        "attempt_id": manga_attempt_id,
                        "recognizer": "manga-ocr",
                        "status": "recognized" if manga_blocks else "empty",
                        "texts": [block.text for block in manga_blocks],
                        "max_confidence": max([block.confidence for block in manga_blocks], default=0.0),
                    }
                ]
            )
            for block in manga_blocks:
                block.metadata = {
                    **block.metadata,
                    "provider": self.name,
                    "recognizer": "manga-ocr",
                    "attempt_id": manga_attempt_id,
                    "text_quality_score": _hybrid_text_quality_score(block.text, block.confidence),
                }
                candidates.append(block)
        except Exception as error:
            attempts.append(
                {
                    "attempt_id": manga_attempt_id,
                    "recognizer": "manga-ocr",
                    "status": "failed",
                    "error_message": str(error),
                }
            )

        selected = _select_hybrid_blocks(candidates)
        if not selected and retry_profile == "mit_quality":
            adaptive_attempt_id = "manga_ocr_adaptive_retry_1"
            try:
                adaptive_blocks = self._manga.recognize_image(
                    image_path=image_path,
                    lang_hint=lang_hint,
                    region=region,
                    polygon=polygon,
                    crop_policy="adaptive",
                    target_text_height=max(target_text_height or 0, 128),
                    padding_ratio=max(padding_ratio or 0.0, 0.35),
                    preprocess="manga_enhance",
                    debug_output_dir=debug_output_dir,
                    metadata={
                        **base_metadata,
                        "attempt_id": adaptive_attempt_id,
                        "recognizer": "manga-ocr-adaptive-retry",
                        "adaptive_retry": True,
                    },
                )
                adaptive_attempts = list(self._manga.last_attempts or [])
                attempts.extend(
                    [
                        {
                            **attempt,
                            "recognizer": "manga-ocr-adaptive-retry",
                            "adaptive_retry": True,
                        }
                        for attempt in adaptive_attempts
                    ]
                    or [
                        {
                            "attempt_id": adaptive_attempt_id,
                            "recognizer": "manga-ocr-adaptive-retry",
                            "status": "recognized" if adaptive_blocks else "empty",
                            "texts": [block.text for block in adaptive_blocks],
                            "max_confidence": max([block.confidence for block in adaptive_blocks], default=0.0),
                        }
                    ]
                )
                for block in adaptive_blocks:
                    block.metadata = {
                        **block.metadata,
                        "provider": self.name,
                        "recognizer": "manga-ocr-adaptive-retry",
                        "attempt_id": adaptive_attempt_id,
                        "adaptive_retry": True,
                        "text_quality_score": _hybrid_text_quality_score(block.text, block.confidence),
                    }
                    candidates.append(block)
            except Exception as error:
                attempts.append(
                    {
                        "attempt_id": adaptive_attempt_id,
                        "recognizer": "manga-ocr-adaptive-retry",
                        "status": "failed",
                        "error_message": str(error),
                    }
                )
            selected = _select_hybrid_blocks(candidates)

        selected_attempt_id = selected[0].metadata.get("attempt_id") if selected else None
        for block in selected:
            block.engine = self.name
            block.metadata = {
                **block.metadata,
                "provider": self.name,
                "selected_attempt_id": selected_attempt_id,
                "attempts": attempts,
            }
        self.last_attempts = attempts
        self.last_selected_attempt_id = str(selected_attempt_id) if selected_attempt_id else None
        return selected


def get_engine(name: str) -> OcrEngine:
    if name == "dummy-static":
        return DummyStaticOcrEngine()
    if name == "dummy-empty":
        return DummyEmptyOcrEngine()
    if name == "dummy":
        return DummyOcrEngine()
    if name == "paddle":
        return PaddleOcrEngine()
    if name == "manga-tiled":
        return MangaTiledOcrEngine()
    if name == "manga-ocr":
        return MangaOcrEngine()
    if name == "mit-48px":
        return Mit48pxAdapterOcrEngine()
    if name == "mit-48px-internal":
        return Mit48pxInternalOcrEngine()
    if name == "mit-manga-hybrid":
        return MitMangaHybridOcrEngine()
    raise OcrEngineError(f"OCR engine '{name}' is not supported.")


def _write_crop(path: str, crop: np.ndarray) -> bool:
    try:
        return bool(cv2.imwrite(path, crop))
    except Exception:
        return False


def _write_debug_crop(debug_output_dir: Optional[Path], metadata: Optional[dict[str, Any]], crop: np.ndarray) -> Optional[Path]:
    if debug_output_dir is None:
        return None
    item_id = str((metadata or {}).get("ocr_batch_item_id") or (metadata or {}).get("source_textline_id") or "crop")
    candidate_index = (metadata or {}).get("candidate_index")
    if candidate_index is not None:
        item_id = f"{item_id}_cand_{candidate_index}"
    safe_id = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in item_id)
    output_path = debug_output_dir / f"{safe_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path if _write_crop(str(output_path), crop) else None


def _manga_ocr_candidate_settings(
    *,
    crop_policy: Optional[str],
    target_text_height: Optional[int],
    padding_ratio: Optional[float],
    preprocess: Optional[str],
) -> list[dict[str, Any]]:
    base = {
        "crop_policy": crop_policy,
        "target_text_height": target_text_height,
        "padding_ratio": padding_ratio,
        "preprocess": preprocess,
    }
    if crop_policy != "adaptive" and os.environ.get("OCR_MANGA_FORCE_ENSEMBLE", "0") != "1":
        return [base]
    heights = []
    for height in [target_text_height, 64, 96]:
        if height and height not in heights:
            heights.append(height)
    if not heights:
        heights = [48, 64, 96]
    preprocessors = []
    for item in [preprocess, "none", "auto"]:
        if item and item not in preprocessors:
            preprocessors.append(item)
    if not preprocessors:
        preprocessors = ["none", "auto"]
    candidates: list[dict[str, Any]] = []
    if crop_policy in (None, "adaptive"):
        for height in [target_text_height or 48, 64, 96]:
            if height and height in heights:
                candidates.append(
                    {
                        "crop_policy": "mit_textline",
                        "target_text_height": height,
                        "padding_ratio": padding_ratio if padding_ratio is not None else 0.04,
                        "preprocess": preprocess or "none",
                    }
                )
    for height in heights:
        for processor in preprocessors:
            candidates.append(
                {
                    "crop_policy": "adaptive",
                    "target_text_height": height,
                    "padding_ratio": padding_ratio if padding_ratio is not None else 0.12,
                    "preprocess": processor,
                }
            )
    return candidates[:6]


def _metadata_orientation(metadata: Optional[dict[str, Any]]) -> Optional[str]:
    if not metadata:
        return None
    for key in ("orientation", "textline_orientation", "detected_orientation"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    bbox = metadata.get("textline_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        width = float(bbox[2])
        height = float(bbox[3])
    except (TypeError, ValueError):
        return None
    return "vertical" if height > width * 1.15 else "horizontal"


def _ocr_candidate_score(text: str) -> float:
    cleaned = text.strip()
    if not cleaned:
        return 0.0
    japanese = sum(1 for char in cleaned if "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff")
    punctuation = sum(1 for char in cleaned if char in "!?！？…。、，．「」『』ー〜～")
    ascii_alnum = sum(1 for char in cleaned if char.isascii() and char.isalnum())
    punctuation_ratio = punctuation / max(1, len(cleaned))
    score = japanese * 2.0 + min(len(cleaned), 16) * 0.1 - ascii_alnum * 0.8
    if punctuation_ratio > 0.65:
        score -= 3.0
    if len(cleaned) <= 1:
        score -= 1.0
    return score


def _dedupe_thresholds(values: list[Optional[float]]) -> list[Optional[float]]:
    seen: set[str] = set()
    thresholds: list[Optional[float]] = []
    for value in values:
        key = "default" if value is None else f"{float(value):.4f}"
        if key in seen:
            continue
        seen.add(key)
        thresholds.append(None if value is None else float(value))
    return thresholds or [None]


def _hybrid_text_quality_score(text: str, confidence: float) -> float:
    cleaned = text.strip()
    if not cleaned:
        return 0.0
    japanese = sum(1 for char in cleaned if "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff")
    latin = sum(1 for char in cleaned if char.isascii() and char.isalpha())
    digits = sum(1 for char in cleaned if char.isascii() and char.isdigit())
    punctuation = sum(1 for char in cleaned if char in "!?！？…。、，．・「」『』ー〜～")
    symbols = len(cleaned) - japanese - latin - digits - punctuation
    score = confidence * 2.0 + japanese * 2.0 + min(latin, 8) * 0.45 + punctuation * 0.2 + min(len(cleaned), 12) * 0.08
    if len(cleaned) <= 1:
        score -= 0.8
    if symbols > 0:
        score -= symbols * 0.7
    if japanese == 0 and latin < 3 and punctuation > 0:
        score -= 1.0
    return float(score)


def _select_hybrid_blocks(blocks: list[OcrTextBlock]) -> list[OcrTextBlock]:
    mit_scored = []
    fallback_scored = []
    for block in blocks:
        score = float(block.metadata.get("text_quality_score", _hybrid_text_quality_score(block.text, block.confidence)) or 0.0)
        recognizer = str(block.metadata.get("recognizer") or block.engine)
        if recognizer == "mit-48px-internal" and _hybrid_mit_block_is_usable(block, score):
            mit_scored.append((_hybrid_selection_score(block, score, prefer_mit=True), block))
        elif recognizer in {"manga-ocr", "manga-ocr-adaptive-retry"} and _hybrid_fallback_block_is_usable(block, score):
            fallback_scored.append((_hybrid_selection_score(block, score, prefer_mit=False), block))
        elif recognizer not in {"mit-48px-internal", "manga-ocr", "manga-ocr-adaptive-retry"} and _hybrid_block_is_usable(block, score):
            fallback_scored.append((_hybrid_selection_score(block, score, prefer_mit=False), block))
    scored = mit_scored or fallback_scored
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    return [scored[0][1]]


def _hybrid_mit_block_is_usable(block: OcrTextBlock, score: float) -> bool:
    text = block.text.strip()
    if not text:
        return False
    confidence = float(block.confidence or 0.0)
    detector_confidence = _metadata_float(block.metadata, "detector_confidence", default=1.0)
    japanese = sum(1 for char in text if "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff")
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    if len(text) <= 1:
        return confidence >= 0.45 and detector_confidence >= 0.65
    if latin >= 3 and confidence >= 0.45:
        return True
    if confidence >= 0.15 and score > 1.5:
        return True
    return confidence >= 0.02 and detector_confidence >= 0.75 and japanese >= 3 and score > 4.0


def _hybrid_fallback_block_is_usable(block: OcrTextBlock, score: float) -> bool:
    text = block.text.strip()
    if not text or len(text) <= 1:
        return False
    detector_confidence = _metadata_float(block.metadata, "detector_confidence", default=0.0)
    crop_ink_ratio = _metadata_float(block.metadata, "crop_ink_ratio", default=0.0)
    punctuation = sum(1 for char in text if char in "!?！？…。、，．・「」『』ー〜～")
    if punctuation / max(1, len(text)) > 0.65:
        return False
    adaptive_retry = block.metadata.get("adaptive_retry") is True or block.metadata.get("recognizer") == "manga-ocr-adaptive-retry"
    if detector_confidence < 0.65 and not (adaptive_retry and crop_ink_ratio >= 0.035 and score > 4.0):
        return False
    if crop_ink_ratio < 0.018 and detector_confidence < 0.85:
        return False
    return score > 4.0


def _hybrid_selection_score(block: OcrTextBlock, score: float, *, prefer_mit: bool) -> float:
    confidence = float(block.confidence or 0.0)
    detector_confidence = _metadata_float(block.metadata, "detector_confidence", default=0.0)
    bonus = 4.0 if prefer_mit else 0.0
    return score + confidence * 3.0 + detector_confidence + bonus


def _hybrid_block_is_usable(block: OcrTextBlock, score: float) -> bool:
    text = block.text.strip()
    if not text:
        return False
    japanese = sum(1 for char in text if "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff")
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    if japanese > 0:
        return score > 0.2
    if latin >= 3:
        return block.confidence >= 0.08 or score > 1.5
    return score > 1.8


def _metadata_float(metadata: dict[str, Any], key: str, *, default: float) -> float:
    try:
        return float(metadata.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _crop_ink_ratio(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    try:
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        return float(np.mean(gray < 110))
    except Exception:
        return 0.0


def _write_debug_crop_metadata(debug_crop_path: Optional[Path], crop_metadata: dict[str, Any], raw_text: str) -> None:
    if debug_crop_path is None:
        return
    metadata_path = debug_crop_path.with_suffix(".json")
    payload = {
        "schema_version": "ocr_crop_debug.v1",
        "crop_image": str(debug_crop_path),
        "raw_text": raw_text,
        "crop": crop_metadata,
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _block_from_adapter_payload(payload: Any, request_metadata: dict[str, Any], engine_name: str = "mit-48px") -> OcrTextBlock:
    if not isinstance(payload, dict):
        raise OcrEngineError("mit-48px adapter block must be an object")
    text = str(payload.get("text", "")).strip()
    if not text:
        raise OcrEngineError("mit-48px adapter block is missing text")
    bbox = BBox([float(value) for value in payload.get("bbox", [0.0, 0.0, 1.0, 1.0])])
    polygon_payload = payload.get("polygon")
    textline_payload = payload.get("textline_polygons", [])
    metadata_payload = payload.get("metadata", {})
    return OcrTextBlock(
        text=text,
        confidence=float(payload.get("confidence", 0.0)),
        bbox=bbox,
        engine=engine_name,
        polygon=Polygon(polygon_payload) if polygon_payload else None,
        textline_polygons=[Polygon(item) for item in textline_payload] if isinstance(textline_payload, list) else [],
        foreground_color=payload.get("foreground_color"),
        background_color=payload.get("background_color"),
        metadata={
            **request_metadata,
            **(metadata_payload if isinstance(metadata_payload, dict) else {}),
            "provider": engine_name,
        },
    )


def _manga_tile_regions(width: int, height: int) -> list[list[float]]:
    # Fast first-pass manga layout regions. Keep this list small: PaddleOCR is
    # heavyweight, and the GUI must not fan out dozens of recognitions per click.
    regions: list[list[float]] = [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.50],
        [0.0, 0.50, 1.0, 0.50],
        [0.70, 0.00, 0.30, 0.42],
        [0.00, 0.68, 0.42, 0.32],
        [0.22, 0.52, 0.34, 0.42],
    ]

    aspect = width / height if height else 1.0
    if aspect < 0.85:
        regions.extend(
            [
                [0.0, 0.0, 1.0, 0.30],
                [0.0, 0.70, 1.0, 0.30],
            ]
        )
    return regions


def _dedupe_text_blocks(blocks: list[OcrTextBlock]) -> list[OcrTextBlock]:
    ranked = sorted(
        blocks,
        key=lambda block: (
            _text_quality(block.text),
            block.confidence,
            block.bbox.root[2] * block.bbox.root[3],
        ),
        reverse=True,
    )
    kept: list[OcrTextBlock] = []
    for block in ranked:
        if _text_quality(block.text) <= 0:
            continue
        if any(_bbox_iou(block.bbox.root, existing.bbox.root) > 0.45 for existing in kept):
            continue
        if any(_same_textish(block.text, existing.text) and _bbox_iou(block.bbox.root, existing.bbox.root) > 0.15 for existing in kept):
            continue
        kept.append(block)
    return sorted(kept, key=lambda block: (block.bbox.root[1], block.bbox.root[0]))


def _text_quality(text: str) -> float:
    cleaned = text.strip()
    if not cleaned:
        return 0.0
    japanese = sum(1 for char in cleaned if "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff")
    punctuation = sum(1 for char in cleaned if char in "!?！？…~ー")
    latin_or_digit = sum(1 for char in cleaned if char.isascii() and char.isalnum())
    if japanese == 0 and punctuation < 2:
        return 0.0
    return japanese * 2.0 + punctuation * 0.5 - latin_or_digit * 0.75 + min(len(cleaned), 8) * 0.1


def _same_textish(left: str, right: str) -> bool:
    left_chars = set(left.strip())
    right_chars = set(right.strip())
    if not left_chars or not right_chars:
        return False
    return len(left_chars & right_chars) / max(len(left_chars), len(right_chars)) > 0.6


def _bbox_iou(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    ix1 = max(lx, rx)
    iy1 = max(ly, ry)
    ix2 = min(lx + lw, rx + rw)
    iy2 = min(ly + lh, ry + rh)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = lw * lh + rw * rh - inter
    return inter / union if union > 0 else 0.0


def _parse_paddle_results(results: object) -> list[tuple[list[list[float]], str, float]]:
    if not results:
        return []

    payload = results[0] if isinstance(results, list) and results else results
    entries: list[tuple[list[list[float]], str, float]] = []

    if isinstance(payload, list):
        for line in payload:
            try:
                bbox_pixels, (text, confidence) = line
                entries.append((bbox_pixels, str(text), float(confidence)))
            except Exception as e:
                logger.warning("Failed to parse OCR line result: %s. Error: %s", line, e)
        return entries

    if isinstance(payload, dict):
        texts = _as_list(payload.get("rec_texts"))
        scores = _as_list(payload.get("rec_scores"))
        raw_polys = payload.get("rec_polys")
        if raw_polys is None:
            raw_polys = payload.get("rec_boxes")
        polys = _as_list(raw_polys)
        for i in range(min(len(texts), len(scores), len(polys))):
            try:
                entries.append((polys[i], str(texts[i]), float(scores[i])))
            except Exception as e:
                logger.warning("Failed to parse structured OCR row idx=%s. Error: %s", i, e)
        return entries

    logger.warning("Unsupported PaddleOCR output type: %s", type(payload))
    return []


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    return list(value) if hasattr(value, "__iter__") and not isinstance(value, str) else []


def _to_text_block(
    bbox_pixels: list[list[float]],
    text: str,
    confidence: float,
    crop_width: int,
    crop_height: int,
    region: Optional[BBox],
    engine_name: str,
) -> Optional[OcrTextBlock]:
    try:
        xs = [float(point[0]) for point in bbox_pixels]
        ys = [float(point[1]) for point in bbox_pixels]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        norm_x = xmin / crop_width if crop_width > 0 else 0.0
        norm_y = ymin / crop_height if crop_height > 0 else 0.0
        norm_w = (xmax - xmin) / crop_width if crop_width > 0 else 0.0
        norm_h = (ymax - ymin) / crop_height if crop_height > 0 else 0.0

        if region is not None:
            rx, ry, rw, rh = region.root
            bbox = [rx + norm_x * rw, ry + norm_y * rh, norm_w * rw, norm_h * rh]
        else:
            bbox = [norm_x, norm_y, norm_w, norm_h]
        x = max(0.0, min(1.0, bbox[0]))
        y = max(0.0, min(1.0, bbox[1]))
        w = max(0.0, min(1.0 - x, bbox[2]))
        h = max(0.0, min(1.0 - y, bbox[3]))
        if w <= 0.001 or h <= 0.001:
            return None
        bbox = [x, y, w, h]

        cleaned = text.strip()
        if not cleaned:
            return None

        return OcrTextBlock(
            text=cleaned,
            confidence=float(confidence),
            bbox=BBox(bbox),
            engine=engine_name,
        )
    except Exception as e:
        logger.warning("Failed to convert OCR result to block. Error: %s", e)
        return None
