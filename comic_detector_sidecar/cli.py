import json
from pathlib import Path
from typing import Optional

import typer

from comic_detector_sidecar.contracts.documents import DetectorOptions
from comic_detector_sidecar.model_manager import prepare_comic_text_detector, prepare_mmocr, prepare_ogkalu_rtdetr, prepare_paddle_layout
from comic_detector_sidecar.pipeline.detect_image import detect_image
from comic_detector_sidecar.pipeline.doctor import run_doctor
from comic_detector_sidecar.utils.jsonl import emit_event

app = typer.Typer(
    name="comic-detector-sidecar",
    help="Standalone comic/manga text detector sidecar CLI",
    no_args_is_help=True,
    add_completion=False,
)


@app.command(name="doctor")
def doctor(
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL doctor event"),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="comic-text-detector model path"),
    adapter_command: Optional[str] = typer.Option(None, "--adapter-command", help="External comic-text-detector adapter command"),
    max_regions: int = typer.Option(24, "--max-regions", help="Maximum regions"),
) -> None:
    payload = run_doctor(DetectorOptions(model_path=model_path, adapter_command=adapter_command, max_regions=max_regions))
    typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


@app.command(name="detect-image")
def detect_image_cmd(
    input_path: Path = typer.Option(..., "--input", help="Input image path"),
    provider: str = typer.Option("mit-ctd", "--provider", help="Detector provider"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output directory"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Manifest output path"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events"),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="comic-text-detector model path"),
    adapter_command: Optional[str] = typer.Option(None, "--adapter-command", help="External comic-text-detector adapter command"),
    max_regions: int = typer.Option(24, "--max-regions", help="Maximum regions"),
    min_confidence: float = typer.Option(0.0, "--min-confidence", help="Minimum region confidence"),
    detection_size: int = typer.Option(1024, "--detection-size", help="Detector input size for model-backed providers"),
    text_threshold: float = typer.Option(0.5, "--text-threshold", help="Text confidence threshold"),
    box_threshold: float = typer.Option(0.45, "--box-threshold", help="Text box confidence threshold"),
    unclip_ratio: float = typer.Option(2.3, "--unclip-ratio", help="Text box expansion ratio"),
    coverage_preset: str = typer.Option("balanced", "--coverage-preset", help="Coverage preset: fast|balanced|quality"),
    nms_threshold: float = typer.Option(0.35, "--nms-threshold", help="Region NMS threshold"),
    min_textline_area_ratio: float = typer.Option(0.000015, "--min-textline-area-ratio", help="Minimum textline area ratio"),
) -> None:
    try:
        detect_image(
            input_path=input_path,
            provider_name=provider,
            options=DetectorOptions(
                model_path=model_path,
                adapter_command=adapter_command,
                max_regions=max_regions,
                min_confidence=min_confidence,
                detection_size=detection_size,
                text_threshold=text_threshold,
                box_threshold=box_threshold,
                unclip_ratio=unclip_ratio,
                coverage_preset=coverage_preset,  # type: ignore[arg-type]
                nms_threshold=nms_threshold,
                min_textline_area_ratio=min_textline_area_ratio,
            ),
            output_dir=output_dir,
            manifest_path=manifest,
            job_id=job_id,
            jsonl=jsonl,
        )
    except Exception:
        raise typer.Exit(1)


@app.command(name="prepare-models")
def prepare_models(
    provider: str = typer.Option("mit-ctd", "--provider", help="Detector provider to prepare"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Prepare job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events"),
    download: bool = typer.Option(True, "--download/--no-download", help="Download missing model files"),
) -> None:
    resolved_job_id = job_id or "comic_detect_prepare"
    try:
        emit_event({"type": "started", "job_id": resolved_job_id, "stage": "comic_detector_prepare", "provider": provider}, jsonl)
        if provider in {"comic-text-detector", "mit-ctd", "ctd-gpl"}:
            status = prepare_comic_text_detector(resolved_job_id, jsonl=jsonl, download=download)
        elif provider == "ogkalu-rtdetr":
            status = prepare_ogkalu_rtdetr(resolved_job_id, jsonl=jsonl, download=download)
        elif provider == "mmocr":
            status = prepare_mmocr(resolved_job_id, jsonl=jsonl, download=download)
        elif provider == "paddle-layout":
            status = prepare_paddle_layout(resolved_job_id, jsonl=jsonl, download=download)
        elif provider == "heuristic":
            status = {
                "name": "heuristic",
                "available": True,
                "installed": True,
                "model_ready": True,
                "needs_download": False,
            }
            emit_event({"type": "model_status", "job_id": resolved_job_id, **status}, jsonl)
        elif provider == "apple-vision":
            status = {
                "name": provider,
                "available": False,
                "installed": False,
                "model_ready": False,
                "needs_download": False,
                "reason": "provider runtime is optional and not prepared by this command",
            }
            emit_event({"type": "model_status", "job_id": resolved_job_id, **status}, jsonl)
        else:
            raise ValueError(f"Comic detector provider '{provider}' is not supported")
        payload = status.to_doctor_payload() if hasattr(status, "to_doctor_payload") else status
        emit_event(
            {
                "type": "done",
                "job_id": resolved_job_id,
                "status": "completed",
                "manifest_path": None,
                "provider": provider,
                "model_status": payload,
            },
            jsonl,
        )
        if not jsonl:
            typer.echo(json.dumps({"provider": provider, "status": "completed", "model_status": payload}, ensure_ascii=False))
    except Exception as error:
        emit_event({"type": "error", "job_id": resolved_job_id, "code": "DETECTOR_MODEL_PREPARE_FAILED", "message": str(error)}, jsonl)
        if not jsonl:
            typer.echo(json.dumps({"error": str(error)}, ensure_ascii=False), err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
