"""JSON report generation for detection results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from streetwatch.detector import Detection


@dataclass
class ImageReport:
    """Structured report for one image."""

    source: str
    timestamp: str
    total_persons: int
    total_cars: int
    detections: list[dict[str, object]] = field(default_factory=list)


def generate_report(
    image_path: Path | str, detections: list[Detection]
) -> ImageReport:
    """Build an ImageReport from a list of detections."""
    image_path = Path(image_path)
    persons = sum(1 for d in detections if d.label == "person")
    cars = sum(1 for d in detections if d.label == "car")

    return ImageReport(
        source=image_path.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_persons=persons,
        total_cars=cars,
        detections=[asdict(d) for d in detections],
    )


def save_report(report: ImageReport, output_path: Path | str) -> Path:
    """Write an ImageReport as JSON and return the file path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    logger.info("Report saved to '{}'", output_path)
    return output_path
