import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from ocr_sidecar import __version__
from ocr_sidecar.benchmark import run_manga_ocr_benchmark
from ocr_sidecar.contracts import (
    BBox,
    BatchItem,
    BatchOptions,
    BatchRequest,
    BatchResultDocument,
    BatchResultItem,
    ManifestWarning,
    OcrManifest,
    OcrManifestImage,
    OcrTextBlock,
    OcrLayoutRegion,
    Polygon,
)
from ocr_sidecar.engines import get_engine
from ocr_sidecar.model_manager import prepare_engine_models, select_best_engine
from ocr_sidecar.protocol import (
    avg_confidence,
    build_blocks_document,
    build_summary,
    doctor_payload,
    dump_json,
    dump_manifest,
    emit_event,
    error_to_code,
    legacy_batch_response,
    legacy_blocks_response,
    make_job_id,
    parse_region,
    print_error,
    read_json,
    SidecarError,
    to_artifact_blocks,
    warning_for_blocks,
)

app = typer.Typer(
    name="ocr-sidecar",
    help="Standalone OCR sidecar CLI",
    no_args_is_help=True,
    add_completion=False,
)


@app.command(name="recognize-image")
def recognize_image(
    legacy_image_path: Optional[Path] = typer.Argument(None, help="Path to an image file"),
    input_path: Optional[Path] = typer.Option(None, "--input", help="Path to an image file"),
    engine: str = typer.Option("dummy-static", "--engine", help="OCR engine name"),
    language_hint: str = typer.Option("auto", "--language-hint", "--lang-hint", help="OCR language hint"),
    region: Optional[str] = typer.Option(None, "--region", help="Normalized x,y,w,h region"),
    output: Optional[Path] = typer.Option(None, "--output", help="Path to write ocr_blocks.json"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Directory for OCR artifacts"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Path to write ocr_manifest.json"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="OCR job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events to stdout"),
    min_confidence: float = typer.Option(0.0, "--min-confidence", help="Warning threshold"),
) -> None:
    resolved_job_id = job_id or make_job_id("job_ocr_image")
    image_path = input_path or legacy_image_path
    try:
        resolved_engine = select_best_engine() if engine == "auto" else engine
        if image_path is None:
            raise SidecarError("INPUT_NOT_FOUND", "--input is required")
        if not image_path.exists():
            raise SidecarError("INPUT_NOT_FOUND", f"Input image not found: {image_path}")

        blocks_path = output or ((output_dir / "ocr_blocks.json") if output_dir else None)
        manifest_path = manifest or ((output_dir / "ocr_manifest.json") if output_dir else None)
        debug_crops_dir = (
            (output_dir / "debug_crops")
            if output_dir and resolved_engine in {"manga-ocr", "mit-48px", "mit-48px-internal", "mit-manga-hybrid"}
            else None
        )

        emit_event(
            {
                "type": "started",
                "job_id": resolved_job_id,
                "sidecar_version": __version__,
                "engine": resolved_engine,
                "requested_engine": engine,
            },
            jsonl,
        )
        emit_event(
            {
                "type": "image_started",
                "job_id": resolved_job_id,
                "image_id": "img_001",
                "image_path": str(image_path),
            },
            jsonl,
        )

        ocr_engine = get_engine(resolved_engine)
        raw_blocks = ocr_engine.recognize_image(
            image_path=image_path,
            lang_hint=language_hint,
            region=parse_region(region),
            debug_output_dir=debug_crops_dir,
            metadata={"ocr_batch_item_id": "img_001"},
        )
        blocks = to_artifact_blocks(raw_blocks, engine=resolved_engine, language_hint=language_hint)
        for block in blocks:
            emit_event(
                {
                    "type": "block_detected",
                    "job_id": resolved_job_id,
                    "image_id": "img_001",
                    "block_id": block.id,
                    "text": block.text,
                    "confidence": block.confidence,
                    "bbox": block.bbox.root,
                    **({"polygon": block.polygon.root} if block.polygon else {}),
                    **({"metadata": block.metadata} if block.metadata else {}),
                },
                jsonl,
            )

        confidence = avg_confidence(blocks)
        warnings = []
        warning = warning_for_blocks(blocks, min_confidence=min_confidence)
        if warning is not None:
            warnings.append(warning)
            emit_event(
                {
                    "type": "warning",
                    "job_id": resolved_job_id,
                    "code": warning.code,
                    "message": warning.message,
                },
                jsonl,
            )

        emit_event(
            {
                "type": "image_completed",
                "job_id": resolved_job_id,
                "image_id": "img_001",
                "block_count": len(blocks),
                "avg_confidence": confidence,
            },
            jsonl,
        )

        blocks_doc = build_blocks_document(
            job_id=resolved_job_id,
            engine=resolved_engine,
            language_hint=language_hint,
            source={"type": "image", "path": str(image_path)},
            blocks=blocks,
        )
        artifacts = {}
        if blocks_path is not None:
            dump_json(blocks_path, blocks_doc.model_dump(mode="json"))
            artifacts["ocr_blocks"] = str(blocks_path)
            emit_event(
                {"type": "artifact", "job_id": resolved_job_id, "kind": "ocr_blocks", "path": str(blocks_path)},
                jsonl,
            )
        if debug_crops_dir is not None and debug_crops_dir.exists():
            artifacts["ocr_debug_crops"] = str(debug_crops_dir)
            emit_event(
                {"type": "artifact", "job_id": resolved_job_id, "kind": "ocr_debug_crops", "path": str(debug_crops_dir)},
                jsonl,
            )

        if manifest_path is not None:
            artifacts["manifest"] = str(manifest_path)
            manifest_doc = OcrManifest(
                job_id=resolved_job_id,
                status="completed",
                engine=resolved_engine,
                language_hint=language_hint,
                source={"type": "image", "path": str(image_path)},
                artifacts=artifacts,
                summary=build_summary(1, len(blocks), [confidence]),
                warnings=warnings,
            )
            dump_manifest(manifest_path, manifest_doc)

        if jsonl:
            emit_event(
                {
                    "type": "done",
                    "job_id": resolved_job_id,
                    "status": "completed",
                    "manifest_path": str(manifest_path) if manifest_path else None,
                },
                True,
            )
        else:
            typer.echo(json.dumps(legacy_blocks_response(resolved_engine, blocks), ensure_ascii=False, separators=(",", ":")))
        _force_exit_after_manga_ocr(resolved_engine)
    except Exception as e:
        print_error(e, job_id=resolved_job_id, jsonl=jsonl)
        raise typer.Exit(1)


@app.command(name="recognize-batch")
def recognize_batch(
    input_path: Optional[Path] = typer.Option(None, "--input", "--input-json", help="Batch request JSON file"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Directory for OCR artifacts"),
    engine: Optional[str] = typer.Option(None, "--engine", help="Override OCR engine name"),
    language_hint: Optional[str] = typer.Option(None, "--language-hint", "--lang-hint", help="Override OCR language hint"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Path to write ocr_manifest.json"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="OCR job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events to stdout"),
    read_stdin: bool = typer.Option(False, "--stdin", help="Read batch request JSON from stdin"),
    min_confidence: float = typer.Option(0.0, "--min-confidence", help="Warning threshold"),
) -> None:
    resolved_job_id = job_id or make_job_id("job_ocr_batch")
    try:
        payload = read_json(input_path, read_stdin)
        request = BatchRequest.model_validate(payload)
        resolved_job_id = job_id or request.job_id or resolved_job_id
        requested_engine = engine or request.engine
        resolved_engine = select_best_engine() if requested_engine == "auto" else requested_engine
        resolved_language = language_hint or request.effective_language_hint()
        manifest_path = manifest or ((output_dir / "ocr_manifest.json") if output_dir else None)
        results_path = (output_dir / "ocr_results.json") if output_dir else None
        blocks_dir = (output_dir / "blocks") if output_dir else None
        debug_crops_dir = (
            (output_dir / "debug_crops")
            if output_dir and resolved_engine in {"manga-ocr", "mit-48px", "mit-48px-internal", "mit-manga-hybrid"}
            else None
        )
        region_debug_crops_dir = (
            (output_dir / "region_debug_crops")
            if output_dir and request.options.use_mocr_merge and request.options.mocr_merge_mode != "off"
            else None
        )
        region_hints_path = (output_dir / "ocr_region_hints.json") if output_dir else None
        region_assignment_debug_path = (output_dir / "ocr_region_assignment_debug.json") if output_dir else None
        debug_report_path = (output_dir / "ocr_textline_quality_report.json") if output_dir else None

        emit_event(
            {
                "type": "started",
                "job_id": resolved_job_id,
                "sidecar_version": __version__,
                "engine": resolved_engine,
                "requested_engine": requested_engine,
            },
            jsonl,
        )

        ocr_engine = get_engine(resolved_engine)
        result_items: list[BatchResultItem] = []
        manifest_images: list[OcrManifestImage] = []
        warnings = []
        all_confidences: list[float] = []
        block_count = 0
        partial_failure = False
        debug_items: list[dict[str, object]] = []

        total = len(request.items)
        for index, item in enumerate(request.items, 1):
            emit_event(
                {
                    "type": "progress",
                    "job_id": resolved_job_id,
                    "stage": "ocr",
                    "progress": index / total if total else 1.0,
                    "message": f"Recognizing image {index}/{total}",
                },
                jsonl,
            )
            emit_event(
                {
                    "type": "image_started",
                    "job_id": resolved_job_id,
                    "image_id": item.id,
                    "image_path": str(item.image_path),
                },
                jsonl,
            )

            item_blocks = []
            error_code = None
            error_message = None
            item_attempts: list[dict[str, object]] = []
            selected_attempt_id = None
            item_status = "empty"
            try:
                raw_blocks, item_attempts, item_status, selected_attempt_id = _recognize_batch_item(
                    ocr_engine=ocr_engine,
                    resolved_engine=resolved_engine,
                    item=item,
                    resolved_language=resolved_language,
                    request_options=request.options,
                    debug_crops_dir=debug_crops_dir,
                )
                item_blocks = to_artifact_blocks(
                    raw_blocks,
                    engine=resolved_engine,
                    language_hint=item.lang_hint or resolved_language,
                )
            except Exception as item_error:
                partial_failure = True
                item_status = "failed"
                error_code = error_to_code(item_error)
                error_message = str(item_error)
                emit_event(
                    {
                        "type": "warning",
                        "job_id": resolved_job_id,
                        "code": "PARTIAL_BATCH_FAILURE",
                        "message": f"Failed to OCR image {item.id}: {item_error}",
                    },
                    jsonl,
                )

            confidence = avg_confidence(item_blocks)
            all_confidences.append(confidence)
            block_count += len(item_blocks)
            for block in item_blocks:
                emit_event(
                    {
                        "type": "block_detected",
                        "job_id": resolved_job_id,
                        "image_id": item.id,
                        "block_id": block.id,
                        "text": block.text,
                        "confidence": block.confidence,
                        "bbox": block.bbox.root,
                        **({"polygon": block.polygon.root} if block.polygon else {}),
                        **({"metadata": block.metadata} if block.metadata else {}),
                    },
                    jsonl,
                )

            item_warning = warning_for_blocks(item_blocks, min_confidence=min_confidence)
            if item_warning is not None:
                warnings.append(item_warning)
                emit_event(
                    {
                        "type": "warning",
                        "job_id": resolved_job_id,
                        "code": item_warning.code,
                        "message": item_warning.message,
                    },
                    jsonl,
                )

            blocks_path = None
            if blocks_dir is not None:
                blocks_path = blocks_dir / f"{item.id}.json"
                blocks_doc = build_blocks_document(
                    job_id=resolved_job_id,
                    engine=resolved_engine,
                    language_hint=item.lang_hint or resolved_language,
                    source={"type": "image", "path": str(item.image_path), "id": item.id},
                    blocks=item_blocks,
                )
                dump_json(blocks_path, blocks_doc.model_dump(mode="json"))

            result_items.append(
                BatchResultItem(
                    id=item.id,
                    image_path=str(item.image_path),
                    blocks=item_blocks,
                    metadata=item.metadata,
                    status=item_status,
                    attempts=item_attempts,
                    selected_attempt_id=selected_attempt_id,
                    error_code=error_code,
                    error_message=error_message,
                )
            )
            manifest_images.append(
                OcrManifestImage(
                    image_id=item.id,
                    image_path=str(item.image_path),
                    blocks_path=str(blocks_path) if blocks_path else None,
                    block_count=len(item_blocks),
                    avg_confidence=confidence,
                    metadata={**item.metadata, "ocr_status": item_status, "selected_attempt_id": selected_attempt_id, "attempt_count": len(item_attempts)},
                    error_code=error_code,
                    error_message=error_message,
                )
            )
            debug_items.append(
                {
                    "id": item.id,
                    "status": item_status,
                    "block_count": len(item_blocks),
                    "recognized": any(block.text.strip() and not block.metadata.get("placeholder") for block in item_blocks),
                    "placeholder": any(bool(block.metadata.get("placeholder")) for block in item_blocks),
                    "attempts": item_attempts,
                    "selected_attempt_id": selected_attempt_id,
                    "metadata": item.metadata,
                }
            )
            emit_event(
                {
                    "type": "image_completed",
                    "job_id": resolved_job_id,
                    "image_id": item.id,
                    "block_count": len(item_blocks),
                    "avg_confidence": confidence,
                },
                jsonl,
            )

        region_hint_report: Optional[dict[str, Any]] = None
        if _should_run_mocr_merge(request):
            region_hint_report = _apply_region_mocr_hints(
                request=request,
                result_items=result_items,
                debug_items=debug_items,
                resolved_language=resolved_language,
                output_dir=output_dir,
                region_debug_crops_dir=region_debug_crops_dir,
                min_confidence=request.options.region_hint_min_confidence,
                job_id=resolved_job_id,
                jsonl=jsonl,
            )

        block_count = sum(len(item.blocks) for item in result_items)
        all_confidences = [avg_confidence(item.blocks) for item in result_items]
        manifest_images = _manifest_images_from_result_items(
            result_items=result_items,
            request_items=request.items,
            blocks_dir=blocks_dir,
        )
        if blocks_dir is not None:
            _rewrite_batch_block_artifacts(
                job_id=resolved_job_id,
                engine=resolved_engine,
                language_hint=resolved_language,
                result_items=result_items,
                blocks_dir=blocks_dir,
            )

        if partial_failure:
            warnings.append(
                ManifestWarning(
                    code="PARTIAL_BATCH_FAILURE",
                    message="One or more OCR batch items failed.",
                )
            )

        result_doc = BatchResultDocument(
            job_id=resolved_job_id,
            engine=resolved_engine,
            language_hint=resolved_language,
            items=result_items,
        )
        artifacts = {}
        if results_path is not None:
            dump_json(results_path, result_doc.model_dump(mode="json"))
            artifacts["ocr_results"] = str(results_path)
            emit_event(
                {"type": "artifact", "job_id": resolved_job_id, "kind": "ocr_results", "path": str(results_path)},
                jsonl,
            )
        if region_hint_report is not None and region_hints_path is not None:
            dump_json(region_hints_path, region_hint_report)
            artifacts["ocr_region_hints"] = str(region_hints_path)
            emit_event(
                {"type": "artifact", "job_id": resolved_job_id, "kind": "ocr_region_hints", "path": str(region_hints_path)},
                jsonl,
            )
        if region_hint_report is not None and region_assignment_debug_path is not None:
            dump_json(region_assignment_debug_path, region_hint_report.get("assignment_debug", {}))
            artifacts["ocr_region_assignment_debug"] = str(region_assignment_debug_path)
            emit_event(
                {
                    "type": "artifact",
                    "job_id": resolved_job_id,
                    "kind": "ocr_region_assignment_debug",
                    "path": str(region_assignment_debug_path),
                },
                jsonl,
            )
        if debug_crops_dir is not None and debug_crops_dir.exists():
            artifacts["ocr_debug_crops"] = str(debug_crops_dir)
            emit_event(
                {"type": "artifact", "job_id": resolved_job_id, "kind": "ocr_debug_crops", "path": str(debug_crops_dir)},
                jsonl,
            )
        if region_debug_crops_dir is not None and region_debug_crops_dir.exists():
            artifacts["ocr_region_debug_crops"] = str(region_debug_crops_dir)
            emit_event(
                {"type": "artifact", "job_id": resolved_job_id, "kind": "ocr_region_debug_crops", "path": str(region_debug_crops_dir)},
                jsonl,
            )
        if debug_report_path is not None and (request.options.debug_level != "none" or debug_crops_dir is not None):
            debug_report = _build_ocr_textline_quality_report(
                job_id=resolved_job_id,
                engine=resolved_engine,
                language_hint=resolved_language,
                request=request,
                debug_items=debug_items,
                debug_crops_dir=debug_crops_dir,
                region_hint_report=region_hint_report,
            )
            dump_json(debug_report_path, debug_report)
            artifacts["ocr_textline_quality_report"] = str(debug_report_path)
            emit_event(
                {"type": "artifact", "job_id": resolved_job_id, "kind": "ocr_debug_report", "path": str(debug_report_path)},
                jsonl,
            )
        if manifest_path is not None:
            artifacts["manifest"] = str(manifest_path)
            manifest_doc = OcrManifest(
                job_id=resolved_job_id,
                status="completed",
                engine=resolved_engine,
                language_hint=resolved_language,
                images=manifest_images,
                artifacts=artifacts,
                summary=build_summary(total, block_count, all_confidences),
                warnings=warnings,
            )
            dump_manifest(manifest_path, manifest_doc)

        if jsonl:
            emit_event(
                {
                    "type": "done",
                    "job_id": resolved_job_id,
                    "status": "completed",
                    "manifest_path": str(manifest_path) if manifest_path else None,
                },
                True,
            )
        else:
            typer.echo(json.dumps(legacy_batch_response(result_doc), ensure_ascii=False, separators=(",", ":")))
        _force_exit_after_manga_ocr(resolved_engine)
    except Exception as e:
        print_error(e, job_id=resolved_job_id, jsonl=jsonl)
        raise typer.Exit(1)


def _should_run_mocr_merge(request: BatchRequest) -> bool:
    if not request.options.use_mocr_merge or request.options.mocr_merge_mode == "off":
        return False
    if not request.items:
        return False
    return bool(request.layout_regions or request.bubbles)


def _apply_region_mocr_hints(
    *,
    request: BatchRequest,
    result_items: list[BatchResultItem],
    debug_items: list[dict[str, object]],
    resolved_language: str,
    output_dir: Optional[Path],
    region_debug_crops_dir: Optional[Path],
    min_confidence: float,
    job_id: str,
    jsonl: bool,
) -> dict[str, Any]:
    item_by_textline_id = {_textline_id_from_result_item(item): item for item in result_items}
    debug_by_id = {str(item.get("id")): item for item in debug_items}
    regions = _dedupe_regions([*request.layout_regions, *request.bubbles])
    hints: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    upgraded_count = 0
    corrected_count = 0

    try:
        mocr_engine = get_engine("manga-ocr")
    except Exception as error:
        return {
            "schema_version": "ocr_region_hints.v1",
            "job_id": job_id,
            "status": "skipped",
            "reason": str(error),
            "summary": {
                "region_hint_count": 0,
                "mocr_upgraded_count": 0,
                "mocr_corrected_count": 0,
                "region_hint_text_coverage": None,
            },
            "region_hints": [],
            "assignment_debug": {"regions": []},
        }

    for region in regions:
        source_ids = _region_source_textline_ids(region)
        if not source_ids:
            continue
        group_items = [item_by_textline_id[source_id] for source_id in source_ids if source_id in item_by_textline_id]
        if not group_items:
            continue
        image_path = Path(group_items[0].image_path)
        hint = _recognize_region_hint(
            engine=mocr_engine,
            image_path=image_path,
            region=region,
            lang_hint=resolved_language,
            debug_output_dir=region_debug_crops_dir,
            padding=request.options.region_crop_padding,
        )
        hints.append(hint)
        assignment = _assign_region_hint_to_items(
            hint=hint,
            region=region,
            group_items=group_items,
            debug_by_id=debug_by_id,
            min_confidence=min_confidence,
            mode=request.options.mocr_merge_mode,
            job_id=job_id,
            jsonl=jsonl,
        )
        assignments.append(assignment)
        upgraded_count += int(assignment.get("upgraded_count", 0) or 0)
        corrected_count += int(assignment.get("corrected_count", 0) or 0)

    hint_text_count = len([hint for hint in hints if str(hint.get("text", "")).strip()])
    return {
        "schema_version": "ocr_region_hints.v1",
        "job_id": job_id,
        "status": "completed",
        "mode": request.options.mocr_merge_mode,
        "source": {"output_dir": str(output_dir) if output_dir else None},
        "summary": {
            "region_hint_count": len(hints),
            "region_hint_text_count": hint_text_count,
            "mocr_upgraded_count": upgraded_count,
            "mocr_corrected_count": corrected_count,
            "region_hint_text_coverage": round(hint_text_count / len(hints), 4) if hints else None,
        },
        "region_hints": hints,
        "assignment_debug": {"regions": assignments},
    }


def _dedupe_regions(regions: list[OcrLayoutRegion]) -> list[OcrLayoutRegion]:
    seen: set[str] = set()
    seen_source_groups: set[tuple[str, ...]] = set()
    result: list[OcrLayoutRegion] = []
    for region in regions:
        if region.id in seen:
            continue
        source_group = tuple(_region_source_textline_ids(region))
        if source_group and source_group in seen_source_groups:
            continue
        seen.add(region.id)
        if source_group:
            seen_source_groups.add(source_group)
        result.append(region)
    return sorted(
        result,
        key=lambda region: (
            int(region.metadata.get("reading_order", region.metadata.get("render_priority", 9999)) or 9999),
            region.id,
        ),
    )


def _recognize_region_hint(
    *,
    engine: Any,
    image_path: Path,
    region: OcrLayoutRegion,
    lang_hint: str,
    debug_output_dir: Optional[Path],
    padding: float,
) -> dict[str, Any]:
    expanded_bbox = BBox(_expand_bbox(region.bbox.root, padding))
    metadata = {
        "ocr_batch_item_id": f"region_{region.id}",
        "source_region_id": region.id,
        "source_textline_ids": _region_source_textline_ids(region),
        "ocr_input_kind": "layout_region",
        "orientation": region.metadata.get("orientation"),
        "mocr_merge_region_kind": region.kind,
    }
    try:
        blocks = engine.recognize_image(
            image_path=image_path,
            lang_hint=lang_hint,
            region=expanded_bbox,
            polygon=region.polygon,
            crop_policy="bbox",
            target_text_height=96,
            padding_ratio=0.04,
            preprocess="auto",
            debug_output_dir=debug_output_dir,
            metadata=metadata,
        )
        attempts = getattr(engine, "last_attempts", None)
        best = blocks[0] if blocks else None
        text = best.text.strip() if best else ""
        confidence = float(best.confidence) if best else 0.0
        return {
            "id": f"hint_{region.id}",
            "region_id": region.id,
            "kind": region.kind,
            "bbox": region.bbox.root,
            "crop_bbox": expanded_bbox.root,
            "source_textline_ids": _region_source_textline_ids(region),
            "text": text,
            "confidence": confidence,
            "text_quality_score": _text_quality_score(text, confidence),
            "status": "recognized" if text else "empty",
            "attempts": attempts if isinstance(attempts, list) else [],
            "metadata": best.metadata if best else {},
        }
    except Exception as error:
        return {
            "id": f"hint_{region.id}",
            "region_id": region.id,
            "kind": region.kind,
            "bbox": region.bbox.root,
            "crop_bbox": expanded_bbox.root,
            "source_textline_ids": _region_source_textline_ids(region),
            "text": "",
            "confidence": 0.0,
            "text_quality_score": 0.0,
            "status": "failed",
            "error_message": str(error),
            "attempts": [],
            "metadata": {},
        }


def _assign_region_hint_to_items(
    *,
    hint: dict[str, Any],
    region: OcrLayoutRegion,
    group_items: list[BatchResultItem],
    debug_by_id: dict[str, dict[str, object]],
    min_confidence: float,
    mode: str,
    job_id: str,
    jsonl: bool,
) -> dict[str, Any]:
    text = str(hint.get("text", "")).strip()
    confidence = float(hint.get("confidence", 0.0) or 0.0)
    assignment = {
        "region_id": region.id,
        "hint_id": hint.get("id"),
        "source_textline_ids": _region_source_textline_ids(region),
        "status": hint.get("status"),
        "hint_text": text,
        "upgraded_ids": [],
        "corrected_ids": [],
        "skipped_reason": None,
    }
    if mode != "fill_empty":
        assignment["skipped_reason"] = "debug_mode"
        return assignment
    if not text:
        assignment["skipped_reason"] = "empty_region_hint"
        return assignment
    if confidence < min_confidence or _text_quality_score(text, confidence) < 1.0:
        assignment["skipped_reason"] = "low_quality_region_hint"
        return assignment

    placeholders = [item for item in group_items if not _item_has_real_text(item)]
    anchors = [item for item in group_items if _item_has_real_text(item)]
    if not placeholders:
        assignment["skipped_reason"] = "no_empty_textlines"
        _attach_region_hint_metadata(group_items, hint, corrected=False)
        return assignment
    if len(placeholders) > 2:
        assignment["skipped_reason"] = "too_many_empty_textlines_for_safe_assignment"
        _attach_region_hint_metadata(group_items, hint, corrected=False)
        return assignment

    residual, matched_anchor_count = _residual_region_text(text, [_item_text(anchor) for anchor in anchors])
    if anchors and matched_anchor_count == 0:
        assignment["skipped_reason"] = "region_hint_not_aligned_with_textline_anchors"
        _attach_region_hint_metadata(group_items, hint, corrected=False)
        return assignment
    if not residual and len(placeholders) == 1 and not anchors:
        residual = text
    if not residual or _text_quality_score(residual, confidence) < 1.0:
        assignment["skipped_reason"] = "no_safe_residual_text"
        _attach_region_hint_metadata(group_items, hint, corrected=False)
        return assignment

    for item in placeholders:
        block = _region_hint_block_for_item(
            item=item,
            text=residual,
            confidence=confidence,
            hint=hint,
        )
        item.blocks = [block]
        item.status = "recognized"
        item.selected_attempt_id = str(hint.get("id"))
        item.metadata = {
            **item.metadata,
            "ocr_status": "recognized",
            "mocr_region_hint_id": hint.get("id"),
            "mocr_region_hint_text": text,
            "mocr_region_upgraded": True,
        }
        debug = debug_by_id.get(item.id)
        if debug is not None:
            debug.update(
                {
                    "status": "recognized",
                    "block_count": 1,
                    "recognized": True,
                    "placeholder": False,
                    "mocr_region_hint_id": hint.get("id"),
                    "mocr_region_hint_text": text,
                    "mocr_region_upgraded": True,
                }
            )
        assignment["upgraded_ids"].append(item.id)
        emit_event(
            {
                "type": "block_detected",
                "job_id": job_id,
                "image_id": item.id,
                "block_id": block.id,
                "text": block.text,
                "confidence": block.confidence,
                "bbox": block.bbox.root,
                **({"polygon": block.polygon.root} if block.polygon else {}),
                "metadata": block.metadata,
            },
            jsonl,
        )
    assignment["upgraded_count"] = len(assignment["upgraded_ids"])
    assignment["corrected_count"] = len(assignment["corrected_ids"])
    return assignment


def _attach_region_hint_metadata(items: list[BatchResultItem], hint: dict[str, Any], *, corrected: bool) -> None:
    for item in items:
        for block in item.blocks:
            block.metadata = {
                **block.metadata,
                "mocr_region_hint_id": hint.get("id"),
                "mocr_region_hint_text": hint.get("text"),
                "mocr_region_corrected": corrected,
            }


def _region_hint_block_for_item(
    *,
    item: BatchResultItem,
    text: str,
    confidence: float,
    hint: dict[str, Any],
) -> Any:
    base = item.blocks[0] if item.blocks else None
    if base is None:
        raise SidecarError("OCR_EXECUTION_FAILED", f"Cannot upgrade OCR item without geometry: {item.id}")
    metadata = {
        **base.metadata,
        "placeholder": False,
        "translatable": True,
        "ocr_status": "recognized",
        "recognizer": "manga-ocr-region-merge",
        "mocr_region_hint_id": hint.get("id"),
        "mocr_region_hint_text": hint.get("text"),
        "mocr_region_hint_confidence": hint.get("confidence"),
        "mocr_region_upgraded": True,
        "text_quality_score": _text_quality_score(text, confidence),
    }
    return base.model_copy(
        update={
            "text": text,
            "confidence": confidence,
            "metadata": metadata,
        },
        deep=True,
    )


def _manifest_images_from_result_items(
    *,
    result_items: list[BatchResultItem],
    request_items: list[BatchItem],
    blocks_dir: Optional[Path],
) -> list[OcrManifestImage]:
    request_by_id = {item.id: item for item in request_items}
    images: list[OcrManifestImage] = []
    for item in result_items:
        source_item = request_by_id.get(item.id)
        blocks_path = blocks_dir / f"{item.id}.json" if blocks_dir is not None else None
        images.append(
            OcrManifestImage(
                image_id=item.id,
                image_path=item.image_path,
                blocks_path=str(blocks_path) if blocks_path else None,
                block_count=len(item.blocks),
                avg_confidence=avg_confidence(item.blocks),
                metadata={
                    **(source_item.metadata if source_item else item.metadata),
                    **item.metadata,
                    "ocr_status": item.status,
                    "selected_attempt_id": item.selected_attempt_id,
                    "attempt_count": len(item.attempts),
                },
                error_code=item.error_code,
                error_message=item.error_message,
            )
        )
    return images


def _rewrite_batch_block_artifacts(
    *,
    job_id: str,
    engine: str,
    language_hint: str,
    result_items: list[BatchResultItem],
    blocks_dir: Path,
) -> None:
    for item in result_items:
        blocks_path = blocks_dir / f"{item.id}.json"
        blocks_doc = build_blocks_document(
            job_id=job_id,
            engine=engine,
            language_hint=language_hint,
            source={"type": "image", "path": item.image_path, "id": item.id},
            blocks=item.blocks,
        )
        dump_json(blocks_path, blocks_doc.model_dump(mode="json"))


def _textline_id_from_result_item(item: BatchResultItem) -> str:
    value = item.metadata.get("source_textline_id")
    return str(value) if value else item.id


def _region_source_textline_ids(region: OcrLayoutRegion) -> list[str]:
    if region.source_textline_ids:
        return region.source_textline_ids
    raw = region.metadata.get("source_textline_ids")
    if isinstance(raw, list):
        return [str(item) for item in raw if isinstance(item, str)]
    return []


def _item_has_real_text(item: BatchResultItem) -> bool:
    for block in item.blocks:
        if block.text.strip() and not block.metadata.get("placeholder"):
            return True
    return False


def _item_text(item: BatchResultItem) -> str:
    texts = [block.text.strip() for block in item.blocks if block.text.strip() and not block.metadata.get("placeholder")]
    return "".join(texts)


def _residual_region_text(text: str, anchors: list[str]) -> tuple[str, int]:
    residual = text.strip()
    matched_anchor_count = 0
    for anchor in anchors:
        anchor = anchor.strip()
        if not anchor:
            continue
        before = residual
        residual = residual.replace(anchor, "")
        if before != residual:
            matched_anchor_count += 1
    return residual.strip(" \n\t\r。、，,.・…！？!?「」『』"), matched_anchor_count


def _expand_bbox(bbox: list[float], pad: float) -> list[float]:
    x, y, w, h = [float(value) for value in bbox]
    left = max(0.0, x - pad)
    top = max(0.0, y - pad)
    right = min(1.0, x + w + pad)
    bottom = min(1.0, y + h + pad)
    return [left, top, max(0.0, right - left), max(0.0, bottom - top)]


def _recognize_batch_item(
    *,
    ocr_engine: Any,
    resolved_engine: str,
    item: BatchItem,
    resolved_language: str,
    request_options: BatchOptions,
    debug_crops_dir: Optional[Path],
) -> tuple[list[OcrTextBlock], list[dict[str, Any]], str, Optional[str]]:
    retry_profile = item.retry_profile or request_options.retry_profile
    preserve_placeholder = item.preserve_placeholder if item.preserve_placeholder is not None else request_options.preserve_empty_textlines
    thresholds = _retry_thresholds(item.probability_threshold, retry_profile, resolved_engine)
    metadata = {
        **item.metadata,
        "ocr_batch_item_id": item.id,
        "retry_profile": retry_profile,
        "quality_preset": request_options.quality_preset,
        "debug_level": request_options.debug_level,
        **({"mit_probability_thresholds": thresholds} if thresholds else {}),
    }
    raw_blocks = ocr_engine.recognize_image(
        image_path=item.image_path,
        lang_hint=item.lang_hint or resolved_language,
        region=item.region,
        polygon=item.polygon,
        crop_policy=item.crop_policy,
        target_text_height=item.target_text_height,
        padding_ratio=item.padding_ratio,
        preprocess=item.preprocess,
        probability_threshold=item.probability_threshold,
        debug_output_dir=debug_crops_dir,
        metadata=metadata,
    )
    attempts = getattr(ocr_engine, "last_attempts", None)
    if not isinstance(attempts, list) or (not attempts and raw_blocks):
        attempts = _attempts_from_blocks(raw_blocks, resolved_engine, item.probability_threshold)
    selected_attempt_id = getattr(ocr_engine, "last_selected_attempt_id", None)
    if not isinstance(selected_attempt_id, str):
        selected_attempt_id = None
    for block in raw_blocks:
        quality = _text_quality_score(str(block.text), block.confidence)
        block.metadata = {
            **block.metadata,
            "ocr_status": "recognized",
            "placeholder": False,
            "source_textline_id": block.metadata.get("source_textline_id") or item.metadata.get("source_textline_id") or item.id,
            "selected_attempt_id": selected_attempt_id,
            "attempt_count": len(attempts),
            "text_quality_score": block.metadata.get("text_quality_score", quality),
        }
    if raw_blocks:
        return raw_blocks, attempts, "recognized", selected_attempt_id
    if not preserve_placeholder:
        return [], attempts, "empty", selected_attempt_id
    placeholder = _placeholder_block(item=item, engine=resolved_engine, metadata=metadata, attempts=attempts)
    return [placeholder], attempts, "placeholder", selected_attempt_id


def _retry_thresholds(threshold: Optional[float], retry_profile: str, engine: str) -> list[Optional[float]]:
    if retry_profile != "mit_quality" or engine not in {"mit-48px-internal", "mit-manga-hybrid"}:
        return [threshold] if threshold is not None else []
    values: list[Optional[float]] = [threshold, 0.05, 0.0]
    seen: set[str] = set()
    result: list[Optional[float]] = []
    for value in values:
        key = "default" if value is None else f"{float(value):.4f}"
        if key in seen:
            continue
        seen.add(key)
        result.append(None if value is None else float(value))
    return result


def _attempts_from_blocks(blocks: list[OcrTextBlock], engine: str, threshold: Optional[float]) -> list[dict[str, Any]]:
    return [
        {
            "attempt_id": "default",
            "recognizer": engine,
            "probability_threshold": threshold,
            "status": "recognized" if blocks else "empty",
            "texts": [block.text for block in blocks],
            "max_confidence": max([block.confidence for block in blocks], default=0.0),
        }
    ]


def _placeholder_block(
    *,
    item: BatchItem,
    engine: str,
    metadata: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> OcrTextBlock:
    polygon = item.polygon
    bbox = item.region or BBox(_bbox_from_polygon(polygon.root) if polygon is not None else [0.0, 0.0, 1.0, 1.0])
    return OcrTextBlock(
        text="",
        confidence=0.0,
        bbox=bbox,
        engine=engine,
        polygon=polygon,
        textline_polygons=[polygon] if polygon is not None else [],
        metadata={
            **metadata,
            "provider": engine,
            "placeholder": True,
            "ocr_status": "empty",
            "translatable": False,
            "source_textline_id": metadata.get("source_textline_id") or item.id,
            "attempt_count": len(attempts),
            "text_quality_score": 0.0,
        },
    )


def _bbox_from_polygon(polygon: list[list[float]]) -> list[float]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    left = max(0.0, min(xs))
    top = max(0.0, min(ys))
    right = min(1.0, max(xs))
    bottom = min(1.0, max(ys))
    return [left, top, max(0.0, right - left), max(0.0, bottom - top)]


def _build_ocr_textline_quality_report(
    *,
    job_id: str,
    engine: str,
    language_hint: str,
    request: BatchRequest,
    debug_items: list[dict[str, object]],
    debug_crops_dir: Optional[Path],
    region_hint_report: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    textline_items = [
        item for item in request.items if item.metadata.get("ocr_input_kind") == "text_line" or item.metadata.get("source_textline_id")
    ]
    denominator = len(textline_items) or len(request.items)
    recognized = [item for item in debug_items if item.get("recognized")]
    placeholders = [item for item in debug_items if item.get("placeholder")]
    geometry_covered = [item for item in debug_items if item.get("recognized") or item.get("placeholder")]
    low_quality = 0
    for item in debug_items:
        for attempt in item.get("attempts", []) if isinstance(item.get("attempts"), list) else []:
            if not isinstance(attempt, dict):
                continue
            texts = attempt.get("texts")
            if isinstance(texts, list):
                for text in texts:
                    if isinstance(text, str) and text.strip() and _text_quality_score(text, float(attempt.get("max_confidence", 0.0) or 0.0)) < 1.0:
                        low_quality += 1
    debug_crop_count = len(list(debug_crops_dir.glob("*.png"))) if debug_crops_dir is not None and debug_crops_dir.exists() else 0
    region_hint_summary = region_hint_report.get("summary", {}) if isinstance(region_hint_report, dict) else {}
    return {
        "schema_version": "ocr_textline_quality_report.v1",
        "job_id": job_id,
        "engine": engine,
        "language_hint": language_hint,
        "summary": {
            "detector_textline_count": len(textline_items),
            "ocr_item_count": len(request.items),
            "recognized_textline_count": len(recognized),
            "placeholder_count": len(placeholders),
            "textline_geometry_coverage": round(len(geometry_covered) / denominator, 4) if denominator else None,
            "recognized_text_coverage": round(len(recognized) / denominator, 4) if denominator else None,
            "empty_after_retry_count": len([item for item in debug_items if item.get("status") in {"empty", "placeholder"}]),
            "low_quality_candidate_count": low_quality,
            "sample_golden_recall": None,
            "debug_crop_count": debug_crop_count,
            "region_hint_count": region_hint_summary.get("region_hint_count", 0),
            "mocr_upgraded_count": region_hint_summary.get("mocr_upgraded_count", 0),
            "mocr_corrected_count": region_hint_summary.get("mocr_corrected_count", 0),
            "region_hint_text_coverage": region_hint_summary.get("region_hint_text_coverage"),
        },
        "region_hints": region_hint_report.get("region_hints", []) if isinstance(region_hint_report, dict) else [],
        "items": debug_items,
    }


def _text_quality_score(text: str, confidence: float) -> float:
    cleaned = text.strip()
    if not cleaned:
        return 0.0
    japanese = sum(1 for char in cleaned if "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff")
    latin = sum(1 for char in cleaned if char.isascii() and char.isalpha())
    punctuation = sum(1 for char in cleaned if char in "!?！？…。、，．・「」『』ー〜～")
    symbols = len(cleaned) - japanese - latin - punctuation
    score = confidence * 2.0 + japanese * 2.0 + min(latin, 8) * 0.45 + punctuation * 0.2 + min(len(cleaned), 12) * 0.08
    if len(cleaned) <= 1:
        score -= 0.8
    if symbols > 0:
        score -= symbols * 0.7
    return float(score)


@app.command(name="doctor")
def doctor(jsonl: bool = typer.Option(False, "--jsonl", help="Emit doctor report as JSONL")) -> None:
    payload = doctor_payload()
    typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


@app.command(name="prepare-models")
def prepare_models(
    engine: str = typer.Option("auto", "--engine", help="OCR engine to prepare: auto|mit-48px-internal|mit-48px|manga-ocr|paddle"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="OCR prepare job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events to stdout"),
    download: bool = typer.Option(True, "--download/--no-download", help="Download missing model files"),
) -> None:
    resolved_job_id = job_id or make_job_id("job_ocr_prepare")
    try:
        selected = prepare_engine_models(engine=engine, job_id=resolved_job_id, jsonl=jsonl, download=download)
        if not jsonl:
            typer.echo(
                json.dumps(
                    {
                        "engine": engine,
                        "selected_engine": selected,
                        "status": "completed",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
    except Exception as e:
        print_error(e, job_id=resolved_job_id, jsonl=jsonl)
        raise typer.Exit(1)


@app.command(name="benchmark-manga")
def benchmark_manga(
    input_path: Path = typer.Option(..., "--input", help="OCR benchmark request JSON"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for benchmark artifacts"),
    engine: Optional[str] = typer.Option(None, "--engine", help="Override OCR engine"),
    language_hint: Optional[str] = typer.Option(None, "--language-hint", "--lang-hint", help="Override language hint"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Benchmark job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events"),
) -> None:
    resolved_job_id = job_id or make_job_id("job_ocr_manga_benchmark")
    try:
        if not input_path.exists():
            raise SidecarError("INPUT_NOT_FOUND", f"Benchmark request not found: {input_path}")
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        emit_event({"type": "started", "job_id": resolved_job_id, "sidecar_version": __version__, "stage": "ocr_benchmark"}, jsonl)
        report = run_manga_ocr_benchmark(
            payload=payload,
            output_dir=output_dir,
            engine=engine,
            language_hint=language_hint,
            job_id=resolved_job_id,
        )
        report_path = output_dir / "ocr_benchmark_report.json"
        emit_event({"type": "artifact", "job_id": resolved_job_id, "kind": "ocr_benchmark_report", "path": str(report_path)}, jsonl)
        emit_event(
            {
                "type": "done",
                "job_id": resolved_job_id,
                "status": "completed",
                "manifest_path": None,
                "result": {"summary": report["summary"], "report": str(report_path)},
            },
            jsonl,
        )
        if not jsonl:
            typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as e:
        print_error(e, job_id=resolved_job_id, jsonl=jsonl)
        raise typer.Exit(1)


def _force_exit_after_manga_ocr(engine: str) -> None:
    if engine not in {"manga-ocr", "mit-manga-hybrid"}:
        return
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    if os.environ.get("OCR_SIDECAR_FORCE_EXIT_AFTER_MANGA_OCR", "1") != "1":
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    app()
