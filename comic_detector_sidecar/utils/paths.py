from pathlib import Path


def ensure_input_image(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input image is not a file: {path}")
