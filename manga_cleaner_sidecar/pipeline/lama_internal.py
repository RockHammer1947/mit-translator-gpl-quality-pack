from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga_cleaner_sidecar.contracts import CleanerConfig, CleanerError
from manga_cleaner_sidecar.pipeline.model_manager import resolve_lama_large_model_path


def clean_with_lama_large_internal(source_image: Path, mask_path: Path, output: Path, config: CleanerConfig) -> None:
    model_path = resolve_lama_large_model_path(config.model_path)
    if not model_path.exists():
        raise CleanerError("MODEL_NOT_FOUND", f"LaMa large model not found: {model_path}")
    try:
        import torch
        from manga_cleaner_sidecar.gpl.manga_translator.inpainting.inpainting_lama_mpe import LamaFourier, load_lama_mpe
    except Exception as error:
        raise CleanerError(
            "CLEANER_PROVIDER_NOT_AVAILABLE",
            f"lama-large-internal requires torch and MIT-derived LaMa modules: {error}",
        ) from error

    source_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if source_bgr is None:
        raise CleanerError("INPUT_NOT_FOUND", f"Input image not readable: {source_image}")
    if mask is None:
        raise CleanerError("MASK_REFINE_FAILED", f"Mask image not readable: {mask_path}")
    if mask.shape[:2] != source_bgr.shape[:2]:
        mask = cv2.resize(mask, (source_bgr.shape[1], source_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    device = _resolve_device(config.device, torch)
    model = load_lama_mpe(str(model_path), device="cpu", use_mpe=False, large_arch=True)
    model.eval()
    if device == "mps":
        model.to(device)
    image_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = _infer_lama_large(
        torch=torch,
        model=model,
        lama_fourier_type=LamaFourier,
        image=image_rgb,
        mask=mask,
        device=device,
        inpainting_size=config.inpainting_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output), result_bgr):
        raise CleanerError("OUTPUT_WRITE_FAILED", f"Failed to write cleaned image: {output}")


def _resolve_device(requested: str | None, torch_module) -> str:
    if requested and requested != "auto":
        return requested
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _infer_lama_large(*, torch, model, lama_fourier_type, image: np.ndarray, mask: np.ndarray, device: str, inpainting_size: int) -> np.ndarray:
    original = np.copy(image)
    mask_original = np.copy(mask)
    mask_original[mask_original < 127] = 0
    mask_original[mask_original >= 127] = 1
    mask_original = mask_original[:, :, None]

    height, width = image.shape[:2]
    if max(height, width) > inpainting_size:
        image = _resize_keep_aspect(image, inpainting_size)
        mask = _resize_keep_aspect(mask, inpainting_size)
    pad = 8
    h, w = image.shape[:2]
    new_h = h if h % pad == 0 else h + (pad - h % pad)
    new_w = w if w % pad == 0 else w + (pad - w % pad)
    if new_h != h or new_w != w:
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if isinstance(model, lama_fourier_type):
        img_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
    else:
        img_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
    mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
    mask_torch[mask_torch < 0.5] = 0
    mask_torch[mask_torch >= 0.5] = 1
    if device == "mps":
        img_torch = img_torch.to(device)
        mask_torch = mask_torch.to(device)
    with torch.no_grad():
        img_torch *= 1 - mask_torch
        result_torch = model(img_torch, mask_torch)
    result_torch = result_torch.to(torch.float32)
    if isinstance(model, lama_fourier_type):
        result = (result_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    else:
        result = ((result_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    if result.shape[0] != height or result.shape[1] != width:
        result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
    return (result * mask_original + original * (1 - mask_original)).astype(np.uint8)


def _resize_keep_aspect(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)

