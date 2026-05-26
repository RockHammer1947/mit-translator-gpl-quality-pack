import json

from typer.testing import CliRunner

from mit_translator_gpl_quality_pack.cli import app


def test_provider_manifest_jsonl_contract() -> None:
    result = CliRunner().invoke(app, ["provider-manifest", "--jsonl"])

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "mit-translator-gpl-quality-pack.v1"
    assert payload["license_class"] == "copyleft_gpl"
    assert payload["manual_enable_required"] is True
    assert {item["id"] for item in payload["capabilities"]} == {
        "mit-ctd",
        "mit-48px-ocr",
        "mit-layout-reference",
    }


def test_doctor_jsonl_reports_all_pack_providers() -> None:
    result = CliRunner().invoke(app, ["doctor", "--jsonl"])

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["type"] == "doctor"
    assert payload["license"] == "GPL-3.0-only"
    assert payload["license_class"] == "copyleft_gpl"
    assert {provider["id"] for provider in payload["providers"]} == {
        "mit-ctd",
        "mit-48px-ocr",
        "mit-layout-reference",
    }
