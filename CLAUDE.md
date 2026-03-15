# Vehicle & Person Detection System

## Project Vision
A real-time vehicle and person detection system powered by YOLOv8. The system processes images and video streams to identify and localize vehicles and people, outputting annotated results with bounding boxes and confidence scores.

## Goals
- Detect vehicles (cars, trucks, buses, motorcycles) and persons in images/video
- Support multiple input sources: image files, video files, webcam streams
- Provide configurable detection parameters (confidence threshold, NMS, classes)
- Output annotated images/video with detection results to the output/ directory

## Architecture Overview

```
src/
├── detector.py      # Core YOLOv8 detection logic
├── config.py        # Configuration management (YAML-based)
├── visualizer.py    # Drawing bounding boxes and labels
└── main.py          # CLI entry point

tests/               # Unit and integration tests
output/              # Detection results (images, videos, logs)
config.yaml          # Runtime configuration
```

## Tech Stack
- **Model**: Ultralytics YOLOv8 (pretrained on COCO)
- **Computer Vision**: OpenCV
- **Configuration**: PyYAML
- **Image Processing**: Pillow
- **Package Manager**: uv

## Development Commands
- `uv sync` — install dependencies
- `uv run python src/main.py` — run the detector
- `uv run pytest` — run tests
