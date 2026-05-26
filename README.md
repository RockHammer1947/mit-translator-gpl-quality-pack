# MIT Translator GPL Quality Pack

This package is an optional GPL-3.0 quality pack for manga translation. It is
not part of the permissive core app and must be installed and enabled manually.

The main app must not import this package. It should discover and run it only
through the stable CLI + JSONL contract.

## Commands

```bash
mit-translator-gpl-quality-pack provider-manifest --jsonl
mit-translator-gpl-quality-pack doctor --jsonl
mit-translator-gpl-quality-pack detect-image --input page.png --output-dir out/detect --job-id job_001 --jsonl
mit-translator-gpl-quality-pack recognize-batch --input batch_request.json --output-dir out/ocr --job-id job_001 --jsonl
mit-translator-gpl-quality-pack merge-textlines --input layout_request.json --output-dir out/layout --job-id job_001 --jsonl
```

## Providers

- `mit-ctd`: MIT-derived comic text detector provider.
- `mit-48px-ocr`: MIT 48px OCR provider.
- `mit-layout-reference`: MIT-style/reference layout provider.

## Delegate Commands

This pack vendors the GPL quality providers and runs them in-process through its
own CLI environment. Advanced users may still override each compatible command:

- `GPL_QUALITY_PACK_COMIC_DETECTOR_CMD`
- `GPL_QUALITY_PACK_OCR_CMD`
- `GPL_QUALITY_PACK_LAYOUT_CMD`

Each override must be an executable command line for a compatible sidecar. The
main application should prefer the top-level `mit-translator-gpl-quality-pack`
commands instead of importing any Python module from this package.

## License

This package is GPL-3.0-only. The main application should display this license
notice before enabling the pack and should never download or bundle it by
default.
