# StreetWatch

Detect **persons** and **cars** in images using [Ultralytics YOLOv8](https://docs.ultralytics.com/).

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
git clone git@github.com:KevinGit1/yoloimage_claudecodetest.git
cd yoloimage_claudecodetest
uv sync
```

This installs all dependencies and the `streetwatch` CLI into a local virtual environment.

## Usage

### Single image

```bash
uv run streetwatch --input photo.jpg
```

### Directory of images

```bash
uv run streetwatch --input images/ --output results/
```

### All options

| Flag       | Default       | Description                              |
|------------|---------------|------------------------------------------|
| `--input`  | *(required)*  | Path to an image file or directory       |
| `--output` | `output/`     | Directory for annotated images & reports |
| `--model`  | `yolov8n.pt`  | YOLO model name or path                  |
| `--conf`   | `0.25`        | Confidence threshold (0.0 - 1.0)         |

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

## Output

For each input image, StreetWatch produces two files in the output directory:

- `<name>_annotated.jpg` -- image with bounding boxes drawn (green = person, blue = car)
- `<name>_report.json` -- structured JSON report

### Example report

```json
{
  "source": "street.jpg",
  "timestamp": "2026-03-16T12:00:00+00:00",
  "total_persons": 3,
  "total_cars": 2,
  "detections": [
    {
      "label": "person",
      "class_id": 0,
      "confidence": 0.92,
      "bbox": [10.0, 20.0, 100.0, 200.0]
    }
  ]
}
```

## Development

### Run tests

```bash
uv run pytest
```

### Lint

```bash
uv run ruff check
```

### Format

```bash
uv run ruff format
```

## Project Structure

```
src/streetwatch/
    __init__.py
    detector.py      # YOLO wrapper, Detection dataclass
    reporter.py      # JSON report generation
    annotator.py     # Bounding box drawing with OpenCV
    cli.py           # CLI entry point
tests/
    test_detector.py
    test_reporter.py
    test_annotator.py
    test_cli.py
output/              # Generated results (gitignored)
```
