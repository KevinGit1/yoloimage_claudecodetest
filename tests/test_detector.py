"""Tests for the detector module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from streetwatch.detector import Detection, Detector, PERSON_CLASS, CAR_CLASS


def _make_mock_result() -> MagicMock:
    """Build a mock ultralytics Result with two boxes."""
    box_person = MagicMock()
    box_person.cls = torch.tensor([PERSON_CLASS])
    box_person.conf = torch.tensor([0.92])
    box_person.xyxy = torch.tensor([[10.0, 20.0, 100.0, 200.0]])

    box_car = MagicMock()
    box_car.cls = torch.tensor([CAR_CLASS])
    box_car.conf = torch.tensor([0.85])
    box_car.xyxy = torch.tensor([[150.0, 60.0, 400.0, 300.0]])

    result = MagicMock()
    result.names = {0: "person", 2: "car"}
    result.boxes = [box_person, box_car]
    return result


@patch("streetwatch.detector.YOLO")
def test_detect_returns_detections(mock_yolo_cls: MagicMock, sample_image: Path) -> None:
    mock_model = MagicMock()
    mock_model.return_value = [_make_mock_result()]
    mock_yolo_cls.return_value = mock_model

    detector = Detector(model_name="yolov8n.pt", conf=0.25)
    detections = detector.detect(sample_image)

    assert len(detections) == 2
    assert all(isinstance(d, Detection) for d in detections)


@patch("streetwatch.detector.YOLO")
def test_detect_labels_correct(mock_yolo_cls: MagicMock, sample_image: Path) -> None:
    mock_model = MagicMock()
    mock_model.return_value = [_make_mock_result()]
    mock_yolo_cls.return_value = mock_model

    detector = Detector()
    detections = detector.detect(sample_image)

    labels = {d.label for d in detections}
    assert labels == {"person", "car"}


@patch("streetwatch.detector.YOLO")
def test_detect_bbox_format(mock_yolo_cls: MagicMock, sample_image: Path) -> None:
    mock_model = MagicMock()
    mock_model.return_value = [_make_mock_result()]
    mock_yolo_cls.return_value = mock_model

    detector = Detector()
    det = detector.detect(sample_image)[0]

    assert len(det.bbox) == 4
    assert all(isinstance(v, float) for v in det.bbox)


@patch("streetwatch.detector.YOLO")
def test_detect_file_not_found(mock_yolo_cls: MagicMock) -> None:
    mock_yolo_cls.return_value = MagicMock()
    detector = Detector()

    with pytest.raises(FileNotFoundError):
        detector.detect(Path("/nonexistent/image.jpg"))


@patch("streetwatch.detector.YOLO")
def test_detect_empty_results(mock_yolo_cls: MagicMock, sample_image: Path) -> None:
    empty_result = MagicMock()
    empty_result.names = {0: "person", 2: "car"}
    empty_result.boxes = []

    mock_model = MagicMock()
    mock_model.return_value = [empty_result]
    mock_yolo_cls.return_value = mock_model

    detector = Detector()
    detections = detector.detect(sample_image)

    assert detections == []
