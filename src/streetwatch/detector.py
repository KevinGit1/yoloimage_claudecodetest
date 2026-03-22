"""YOLO-based object detector for persons and cars."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from ultralytics import YOLO


PERSON_CLASS: int = 0
CAR_CLASS: int = 2
TARGET_CLASSES: list[int] = [PERSON_CLASS, CAR_CLASS]


@dataclass(frozen=True)
class Detection:
    """A single detected object."""

    label: str
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) xyxy format


class Detector:
    """Wraps ultralytics YOLO for person/car detection."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25) -> None:
        self._model = YOLO(model_name)
        self._conf = conf
        logger.info("Loaded model '{}' with conf={}", model_name, conf)

    def detect(self, image_path: Path | str) -> list[Detection]:
        """Run detection on a single image and return filtered results."""
        # Runs YOLO inference and filters results to target classes
        # Filter for COCO classes 0 (person) and 2 (car) only
        # Delegates to ultralytics YOLO with pre-configured confidence and class filter
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self._model(
            source=str(image_path),
            conf=self._conf,
            classes=TARGET_CLASSES,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append(
                    Detection(
                        label=names[class_id],
                        class_id=class_id,
                        confidence=float(box.conf[0]),
                        bbox=tuple(float(v) for v in box.xyxy[0]),
                    )
                )

        logger.info(
            "Detected {} objects in '{}'", len(detections), image_path.name
        )
        return detections
