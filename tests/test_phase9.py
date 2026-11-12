"""Phase 9: fact sheet, figure sizes, one-command rebuild, presentation assets."""

import importlib.util
import json

import pytest

from aicidentia import reporting
from aicidentia.config import PROJECT_ROOT


def _run_all():
    spec = importlib.util.spec_from_file_location(
        "run_all", PROJECT_ROOT / "scripts" / "run_all.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_paths_into_registry_values(monkeypatch):
    store = {
        "a.b": {"x": {"y": 2}},
        "t": [
            {"model": "m1", "setting": "cv", "f1": 0.5},
            {"model": "m1", "setting": "time", "f1": 0.4},
        ],
    }
    monkeypatch.setattr(reporting.registry, "get", lambda key: store[key])
    assert reporting.resolve("a.b:x.y") == 2
    assert reporting.resolve("t:[model=m1;setting=time].f1") == 0.4
    with pytest.raises(StopIteration):
        reporting.resolve("t:[model=zz].f1")


def test_report_facts_have_no_missing_values(tmp_path):
    path = tmp_path / "facts.md"
    n = reporting.write_report_facts(path)
    text = path.read_text(encoding="utf-8")
    assert n > 100 and "(missing)" not in text
    for title, _ in reporting.SECTIONS:
        assert f"## {title}" in text


def test_figures_are_ieee_width_png_and_svg():
    from PIL import Image

    pngs = sorted((PROJECT_ROOT / "reports" / "figures").glob("fig_*.png"))
    assert len(pngs) >= 30
    for png in pngs:
        width_in = Image.open(png).size[0] / 300
        assert round(width_in, 2) in (3.5, 7.16), png.name
        assert png.with_suffix(".svg").exists(), png.name


def test_raw_data_match_the_manifest():
    assert "match docs/data_manifest.md" in _run_all().verify_raw()


def test_key_findings_reference_existing_keys():
    from aicidentia import registry

    findings = registry.get("story.key_findings")
    assert len(findings) == 5
    results = registry.load_results()
    for f in findings:
        for key in (k.strip() for k in f["key"].split(";")):
            assert key in results, key
        assert f["limitation"] and f["page"]


def test_presentation_assets_exist():
    svg = (PROJECT_ROOT / "docs" / "methods_overview.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg") and "RQ4" in svg
    shots = sorted((PROJECT_ROOT / "reports" / "figures" / "dashboard").glob("*.png"))
    assert len(shots) == 7
    run_log = json.loads(
        (PROJECT_ROOT / "reports" / "colab" / "rq3_distilbert_run.json").read_text(encoding="utf-8")
    )
    assert run_log["protocol"]["epochs"] == 4
