"""Draw bounding boxes on images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from streetwatch.detector import PERSON_CLASS, Detection

# BGR colours
COLOR_PERSON: tuple[int, int, int] = (0, 255, 0)   # green
COLOR_CAR: tuple[int, int, int] = (255, 0, 0)       # blue
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float = 0.6
THICKNESS: int = 2


def annotate_image(
    image_path: Path | str, detections: list[Detection]
) -> np.ndarray:
    """Load an image and draw coloured bounding boxes for each detection."""
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for det in detections:
        color = COLOR_PERSON if det.class_id == PERSON_CLASS else COLOR_CAR
        x1, y1, x2, y2 = (int(v) for v in det.bbox)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)

        label_text = f"{det.label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            img, label_text, (x1, y1 - 4),
            FONT, FONT_SCALE, (255, 255, 255), THICKNESS,
        )

    logger.info(
        "Annotated '{}' with {} detections", image_path.name, len(detections)
    )
    return img


def save_annotated(image: np.ndarray, output_path: Path | str) -> Path:
    """Write an annotated image to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info("Annotated image saved to '{}'", output_path)
    return output_path
