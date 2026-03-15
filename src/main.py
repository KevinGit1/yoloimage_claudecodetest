import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

TARGET_CLASSES = {0: "person", 2: "car"}
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def parse_args():
    parser = argparse.ArgumentParser(description="Detect persons and cars in an image using YOLOv8.")
    parser.add_argument("image", type=Path, help="Path to the input image file")
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum confidence threshold (default: 0.25)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image.exists():
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    model = YOLO("yolov8n.pt")
    results = model(args.image, conf=args.confidence)
    result = results[0]

    boxes = result.boxes
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id not in TARGET_CLASSES:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append((TARGET_CLASSES[cls_id], conf, x1, y1, x2, y2))

    if not detections:
        print("No persons or cars detected.")
    else:
        print(f"Found {len(detections)} detection(s):\n")
        for name, conf, x1, y1, x2, y2 in detections:
            print(f"  {name:>6s}  conf={conf:.2f}  bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.image.stem}_detected{args.image.suffix}"
    annotated = result.plot()

    import cv2
    cv2.imwrite(str(output_path), annotated)
    print(f"\nAnnotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
