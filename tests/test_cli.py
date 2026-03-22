"""Tests for the CLI module."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from streetwatch.cli import _collect_images, parse_args, run
from streetwatch.detector import Detection


def test_parse_args_required_input() -> None:
    args = parse_args(["--input", "photo.jpg"])

    assert args.input == Path("photo.jpg")
    assert args.output == Path("output")


def test_parse_args_all_options() -> None:
    args = parse_args([
        "--input", "imgs/",
        "--output", "results/",
        "--model", "yolov8s.pt",
        "--conf", "0.5",
    ])

    assert args.input == Path("imgs/")
    assert args.output == Path("results/")
    assert args.model == "yolov8s.pt"
    assert args.conf == 0.5


def test_collect_images_single_file(sample_image: Path) -> None:
    images = _collect_images(sample_image)

    assert images == [sample_image]


def test_collect_images_directory(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    for name in ["a.jpg", "b.png", "c.txt"]:
        p = tmp_path / name
        if name.endswith(".txt"):
            p.write_text("not an image")
        else:
            cv2.imwrite(str(p), np.zeros((10, 10, 3), dtype=np.uint8))

    images = _collect_images(tmp_path)

    assert len(images) == 2
    assert all(p.suffix in {".jpg", ".png"} for p in images)


def test_collect_images_not_found() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        _collect_images(Path("/no/such/path"))


@patch("streetwatch.cli.Detector")
def test_run_creates_outputs(
    mock_detector_cls: MagicMock, sample_image: Path, tmp_path: Path
) -> None:
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [
        Detection(label="person", class_id=0, confidence=0.9, bbox=(10.0, 20.0, 50.0, 80.0)),
    ]
    mock_detector_cls.return_value = mock_detector

    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        input=sample_image,
        output=output_dir,
        model="yolov8n.pt",
        conf=0.25,
    )
    run(args)

    stem = sample_image.stem
    assert (output_dir / f"{stem}_annotated.jpg").exists()
    assert (output_dir / f"{stem}_report.json").exists()
