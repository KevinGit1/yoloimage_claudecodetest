"""CLI entry point for StreetWatch."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from streetwatch.annotator import annotate_image, save_annotated
from streetwatch.detector import Detector
from streetwatch.reporter import generate_report, save_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="streetwatch",
        description="Detect persons and cars in images using YOLOv8.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to an image file or directory of images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory for annotated images and reports (default: output/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model name or path (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    return parser.parse_args(argv)


def _collect_images(input_path: Path) -> list[Path]:
    """Resolve input path to a list of image files."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in extensions
        )
    raise FileNotFoundError(f"Input path not found: {input_path}")


def run(args: argparse.Namespace) -> None:
    """Execute the detection pipeline."""
    images = _collect_images(args.input)
    if not images:
        logger.warning("No images found at '{}'", args.input)
        return

    logger.info("Processing {} image(s)", len(images))
    detector = Detector(model_name=args.model, conf=args.conf)

    for image_path in images:
        detections = detector.detect(image_path)

        stem = image_path.stem
        annotated = annotate_image(image_path, detections)
        save_annotated(annotated, args.output / f"{stem}_annotated.jpg")

        report = generate_report(image_path, detections)
        save_report(report, args.output / f"{stem}_report.json")

    logger.info("Done. Results saved to '{}'", args.output)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    run(args)
