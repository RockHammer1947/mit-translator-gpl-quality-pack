# GPL-Derived Inpainting Notice

This directory contains selected GPL-3.0-derived code adapted from the local
`对标项目/manga-image-translator-main` reference project.

Source module:

- `manga_translator/inpainting/inpainting_lama_mpe.py`

The surrounding sidecar keeps the process boundary explicit: model execution
runs in this standalone CLI sidecar, emits JSONL, writes artifacts to disk, and
then exits so the operating system can reclaim Python/PyTorch/MPS memory.

Model weights are not committed. Use:

```bash
uv run manga-cleaner-sidecar prepare-models --provider lama-large-internal --jsonl
```

