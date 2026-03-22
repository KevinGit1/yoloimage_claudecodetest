"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from streetwatch.detector import Detection


@pytest.fixture()
def sample_detections() -> list[Detection]:
    """A fixed set of detections for testing."""
    return [
        Detection(label="person", class_id=0, confidence=0.92, bbox=(10.0, 20.0, 100.0, 200.0)),
        Detection(label="car", class_id=2, confidence=0.85, bbox=(150.0, 60.0, 400.0, 300.0)),
        Detection(label="person", class_id=0, confidence=0.71, bbox=(200.0, 30.0, 280.0, 190.0)),
    ]


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """Create a small dummy image on disk."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (128, 128, 128)
    path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(path), img)
    return path
