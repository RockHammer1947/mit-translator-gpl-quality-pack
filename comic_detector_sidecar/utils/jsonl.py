import json
import sys
from typing import Any

from comic_detector_sidecar.contracts.documents import SCHEMA_VERSION


def emit_event(event: dict[str, Any], jsonl: bool) -> None:
    if not jsonl:
        return
    event.setdefault("schema_version", SCHEMA_VERSION)
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")))


def log(message: str) -> None:
    print(message, file=sys.stderr)
