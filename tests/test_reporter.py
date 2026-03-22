"""Tests for the reporter module."""

from __future__ import annotations

import json
from pathlib import Path

from streetwatch.detector import Detection
from streetwatch.reporter import generate_report, save_report


def test_generate_report_counts(sample_detections: list[Detection]) -> None:
    report = generate_report("street.jpg", sample_detections)

    assert report.total_persons == 2
    assert report.total_cars == 1


def test_generate_report_source(sample_detections: list[Detection]) -> None:
    report = generate_report("/some/path/street.jpg", sample_detections)

    assert report.source == "street.jpg"


def test_generate_report_has_timestamp(sample_detections: list[Detection]) -> None:
    report = generate_report("street.jpg", sample_detections)

    assert report.timestamp  # non-empty
    assert "T" in report.timestamp  # ISO format


def test_generate_report_detections_serialized(
    sample_detections: list[Detection],
) -> None:
    report = generate_report("street.jpg", sample_detections)

    assert len(report.detections) == 3
    assert report.detections[0]["label"] == "person"
    assert "bbox" in report.detections[0]


def test_generate_report_empty() -> None:
    report = generate_report("empty.jpg", [])

    assert report.total_persons == 0
    assert report.total_cars == 0
    assert report.detections == []


def test_save_report_creates_file(
    tmp_path: Path, sample_detections: list[Detection]
) -> None:
    report = generate_report("street.jpg", sample_detections)
    out = save_report(report, tmp_path / "report.json")

    assert out.exists()
    data = json.loads(out.read_text())
    assert data["total_persons"] == 2
    assert data["total_cars"] == 1


def test_save_report_creates_parent_dirs(
    tmp_path: Path, sample_detections: list[Detection]
) -> None:
    report = generate_report("street.jpg", sample_detections)
    nested = tmp_path / "a" / "b" / "report.json"
    out = save_report(report, nested)

    assert out.exists()
