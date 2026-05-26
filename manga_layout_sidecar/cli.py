from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Optional

import typer

from manga_layout_sidecar import __version__
from manga_layout_sidecar.contracts import SCHEMA_VERSION
from manga_layout_sidecar.pipeline.merge_textlines import merge_textlines

app = typer.Typer(
    name="manga-layout-sidecar",
    help="Clean-room manga textline merge and layout sidecar",
    no_args_is_help=True,
    add_completion=False,
)


def emit(event: dict, jsonl: bool) -> None:
    if jsonl:
        print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)
    else:
        typer.echo(json.dumps(event, ensure_ascii=False, indent=2))


@app.command("doctor")
def doctor(jsonl: bool = typer.Option(False, "--jsonl")) -> None:
    emit(
        {
            "type": "doctor",
            "job_id": "doctor",
            "schema_version": SCHEMA_VERSION,
            "sidecar_version": __version__,
            "python_version": platform.python_version(),
            "available": True,
            "license": "project",
            "provider": "native_layout",
            "compatibility_modes": ["mit_textline_merge"],
        },
        jsonl,
    )


@app.command("merge-textlines")
def merge_textlines_cmd(
    input_path: Path = typer.Option(..., "--input", help="manga_layout_request.v1 JSON file"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output directory"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Manifest output path"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Job id"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit JSONL events"),
) -> None:
    resolved_job_id = job_id or "manga_layout"
    try:
        emit(
            {
                "type": "started",
                "job_id": resolved_job_id,
                "schema_version": SCHEMA_VERSION,
                "sidecar_version": __version__,
                "provider": "native_layout",
            },
            jsonl,
        )
        result = merge_textlines(
            input_path=input_path,
            output_dir=output_dir,
            manifest_path=manifest,
            job_id=resolved_job_id,
            jsonl=jsonl,
            emit=lambda event: emit(event, jsonl),
        )
        if jsonl:
            for kind, path in result["artifacts"].items():
                emit({"type": "artifact", "job_id": resolved_job_id, "kind": kind, "path": path}, True)
            emit(
                {
                    "type": "done",
                    "job_id": resolved_job_id,
                    "status": "completed",
                    "manifest_path": result["artifacts"]["manifest"],
                },
                True,
            )
        else:
            typer.echo(json.dumps(result["document"], ensure_ascii=False, indent=2))
    except Exception as error:
        emit(
            {
                "type": "error",
                "job_id": resolved_job_id,
                "schema_version": SCHEMA_VERSION,
                "code": "LAYOUT_MERGE_FAILED",
                "message": str(error),
            },
            jsonl,
        )
        if not jsonl:
            typer.echo(str(error), err=True)
        else:
            print(str(error), file=sys.stderr)
        raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
