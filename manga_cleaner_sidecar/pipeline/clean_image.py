from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2

from manga_cleaner_sidecar.contracts import CleanImageRequest, CleanerConfig, CleanerError
from manga_cleaner_sidecar.pipeline.lama_internal import clean_with_lama_large_internal
from manga_cleaner_sidecar.pipeline.mask_refinement import (
    build_refined_mask,
    save_mask,
    save_mask_debug_overlay,
)
from manga_cleaner_sidecar.pipeline.quality import build_inpaint_quality_report


def clean_image(request: CleanImageRequest) -> dict[str, Any]:
    started_at = time.perf_counter()
    if not request.source_image.exists():
        raise CleanerError("INPUT_NOT_FOUND", f"Input image not found: {request.source_image}")
    if request.raw_mask_image is not None and not request.raw_mask_image.exists():
        raise CleanerError("INPUT_NOT_FOUND", f"Raw mask image not found: {request.raw_mask_image}")
    if request.detector_refined_mask_image is not None and not request.detector_refined_mask_image.exists():
        raise CleanerError("INPUT_NOT_FOUND", f"Detector refined mask image not found: {request.detector_refined_mask_image}")

    refined = build_refined_mask(
        request.source_image,
        request.raw_mask_image,
        request.detector_refined_mask_image,
        request.blocks,
        request.config,
    )
    save_mask(request.mask_output, refined.mask)
    if request.refined_mask_output is not None:
        save_mask(request.refined_mask_output, refined.mask)
    if request.mask_debug_output is not None:
        save_mask_debug_overlay(request.source_image, refined.mask, request.mask_debug_output)

    cleaned_output = request.cleaned_output
    cleaned_output.parent.mkdir(parents=True, exist_ok=True)
    provider = request.config.provider
    if provider == "none":
        shutil.copyfile(request.source_image, cleaned_output)
    elif provider == "telea":
        _clean_with_telea(request.source_image, request.mask_output, cleaned_output, request.config)
    elif provider == "lama-large":
        _clean_with_lama_external(request.source_image, request.mask_output, cleaned_output, request.config)
    elif provider == "lama-large-internal":
        clean_with_lama_large_internal(request.source_image, request.mask_output, cleaned_output, request.config)
    else:
        raise CleanerError("CLEANER_PROVIDER_NOT_FOUND", f"Unknown cleaner provider: {provider}")

    quality_report = build_inpaint_quality_report(
        job_id=request.job_id,
        provider=provider,
        source_image=request.source_image,
        cleaned_image=cleaned_output,
        mask_image=request.mask_output,
        started_at=started_at,
        mask_refinement_stats=refined.stats,
    )

    return {
        "provider": provider,
        "mask_pixels": int(refined.stats["refined_mask_pixels"]),
        "mask_refinement": refined.stats,
        "cleaned_output": str(cleaned_output),
        "inpaint_quality_report": quality_report,
    }


def build_mask_only(request: CleanImageRequest) -> dict[str, Any]:
    if not request.source_image.exists():
        raise CleanerError("INPUT_NOT_FOUND", f"Input image not found: {request.source_image}")
    refined = build_refined_mask(
        request.source_image,
        request.raw_mask_image,
        request.detector_refined_mask_image,
        request.blocks,
        request.config,
    )
    save_mask(request.mask_output, refined.mask)
    if request.refined_mask_output is not None:
        save_mask(request.refined_mask_output, refined.mask)
    if request.mask_debug_output is not None:
        save_mask_debug_overlay(request.source_image, refined.mask, request.mask_debug_output)
    return {
        "provider": "mask-only",
        "mask_pixels": int(refined.stats["refined_mask_pixels"]),
        "mask_refinement": refined.stats,
    }


def _clean_with_telea(source_image: Path, mask_path: Path, output: Path, config: CleanerConfig) -> None:
    source = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise CleanerError("INPUT_NOT_FOUND", f"Input image not readable: {source_image}")
    if mask is None:
        raise CleanerError("MASK_REFINE_FAILED", f"Mask image not readable: {mask_path}")
    if mask.shape[:2] != source.shape[:2]:
        mask = cv2.resize(mask, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST)
    cleaned = cv2.inpaint(source, mask, float(config.inpaint_radius), cv2.INPAINT_TELEA)
    if not cv2.imwrite(str(output), cleaned):
        raise CleanerError("OUTPUT_WRITE_FAILED", f"Failed to write cleaned image: {output}")


def _clean_with_lama_external(source_image: Path, mask_path: Path, output: Path, config: CleanerConfig) -> None:
    command = config.lama_command or os.environ.get("MANGA_CLEANER_LAMA_CMD")
    if not command:
        raise CleanerError(
            "CLEANER_PROVIDER_NOT_AVAILABLE",
            "lama-large requires --lama-command or MANGA_CLEANER_LAMA_CMD in this clean-room adapter",
        )
    argv = shlex.split(command)
    if not shutil.which(argv[0]):
        raise CleanerError("CLEANER_PROVIDER_NOT_AVAILABLE", f"LaMa command not found: {argv[0]}")
    if config.model_path is not None and not config.model_path.exists():
        raise CleanerError("MODEL_NOT_FOUND", f"LaMa model path not found: {config.model_path}")
    full_argv = [
        *argv,
        "--input",
        str(source_image),
        "--mask",
        str(mask_path),
        "--output",
        str(output),
        "--inpainting-size",
        str(config.inpainting_size),
        "--precision",
        config.precision,
    ]
    if config.model_path is not None:
        full_argv.extend(["--model-path", str(config.model_path)])
    if config.device:
        full_argv.extend(["--device", config.device])
    completed = subprocess.run(full_argv, capture_output=True, text=True, check=False)
    if completed.stdout:
        sys.stderr.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise CleanerError(
            "CLEANER_EXECUTION_FAILED",
            f"LaMa adapter failed with exit code {completed.returncode}",
        )
    if not output.exists():
        raise CleanerError("OUTPUT_WRITE_FAILED", f"LaMa adapter did not write output: {output}")
