"""Tests for the annotator module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from streetwatch.annotator import annotate_image, save_annotated
from streetwatch.detector import Detection


def test_annotate_image_returns_ndarray(
    sample_image: Path, sample_detections: list[Detection]
) -> None:
    result = annotate_image(sample_image, sample_detections)

    assert isinstance(result, np.ndarray)
    assert result.shape[2] == 3  # BGR


def test_annotate_image_preserves_dimensions(
    sample_image: Path, sample_detections: list[Detection]
) -> None:
    original = cv2.imread(str(sample_image))
    annotated = annotate_image(sample_image, sample_detections)

    assert annotated.shape == original.shape


def test_annotate_image_modifies_pixels(
    sample_image: Path, sample_detections: list[Detection]
) -> None:
    original = cv2.imread(str(sample_image))
    annotated = annotate_image(sample_image, sample_detections)

    assert not np.array_equal(original, annotated)


def test_annotate_image_no_detections(sample_image: Path) -> None:
    original = cv2.imread(str(sample_image))
    annotated = annotate_image(sample_image, [])

    np.testing.assert_array_equal(original, annotated)


def test_annotate_image_file_not_found() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        annotate_image(Path("/nonexistent/img.jpg"), [])


def test_save_annotated_creates_file(tmp_path: Path) -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = save_annotated(img, tmp_path / "out.jpg")

    assert out.exists()
    reloaded = cv2.imread(str(out))
    assert reloaded is not None


def test_save_annotated_creates_parent_dirs(tmp_path: Path) -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    nested = tmp_path / "x" / "y" / "out.jpg"
    out = save_annotated(img, nested)

    assert out.exists()
