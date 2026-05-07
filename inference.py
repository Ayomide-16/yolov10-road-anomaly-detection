"""
=============================================================================
Group 8 — MCE 415: YOLOv10 Road Anomaly Detection
inference.py  |  Command-Line Inference Script
=============================================================================
Usage
-----
  # Run inference on a single image
  python inference.py --weights weights/lr_0.001/best.pt --image road.jpg

  # Run on a folder of images
  python inference.py --weights weights/lr_0.001/best.pt --image ./test_images/

  # Adjust confidence and IoU thresholds
  python inference.py --weights weights/lr_0.001/best.pt --image road.jpg --conf 0.4 --iou 0.5

  # Save results to a custom output directory
  python inference.py --weights weights/lr_0.001/best.pt --image road.jpg --output ./my_results/

  # Suppress display (just save results)
  python inference.py --weights weights/lr_0.001/best.pt --image road.jpg --no-show

Weight Files
------------
  Place your trained .pt files in the weights/ folder alongside this script.
  Expected structure (matching your training runs):

    weights/
      lr_1e-05/best.pt
      lr_0.0001/best.pt
      lr_0.001/best.pt
      lr_0.01/best.pt
      lr_0.1/best.pt

  Or simply pass the full path to any .pt file directly.

Requirements
------------
  pip install -r requirements.txt
=============================================================================
"""

import argparse
import sys
import os
import time
from pathlib import Path

# ── Dependency check ─────────────────────────────────────────────────────────
def check_dependencies():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        from ultralytics import YOLO
    except ImportError:
        missing.append("ultralytics")
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    if missing:
        print(f"\n[ERROR] Missing required packages: {', '.join(missing)}")
        print("       Run:  pip install -r requirements.txt\n")
        sys.exit(1)

check_dependencies()

import torch
import cv2
import numpy as np
from ultralytics import YOLO

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Pothole", "Speedbump", "Crack"]
CLASS_COLORS = {           # BGR for OpenCV drawing
    "Pothole"   : (0,  82, 212),   # blue
    "Speedbump" : (0, 160,  50),   # green
    "Crack"     : (0, 100, 220),   # orange-red
}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      Group 8 — MCE 415  |  YOLOv10 Road Anomaly Detection   ║
║              Command-Line Inference Script                   ║
╚══════════════════════════════════════════════════════════════╝
  Classes  : Pothole  |  Speedbump  |  Crack
  Model    : YOLOv10-M (fine-tuned on road anomaly dataset)
"""


# ── Argument Parser ───────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description="YOLOv10 Road Anomaly Detection — Group 8 MCE 415",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--weights", "-w",
        type=str,
        required=True,
        help="Path to trained .pt weights file  (e.g. weights/lr_0.001/best.pt)",
    )
    p.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to an image file OR a folder of images",
    )
    p.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold  [default: 0.25]",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS  [default: 0.45]",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size  [default: 640]",
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default="inference_results",
        help="Output directory for saved results  [default: inference_results/]",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display images (just save results)",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results (just display)",
    )
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────
def find_images(path: Path) -> list[Path]:
    """Return list of image paths from a file or directory."""
    if path.is_file():
        if path.suffix.lower() in IMG_EXTENSIONS:
            return [path]
        else:
            print(f"[WARNING] '{path}' is not a recognised image format.")
            return []
    elif path.is_dir():
        imgs = sorted([
            f for f in path.iterdir()
            if f.suffix.lower() in IMG_EXTENSIONS
        ])
        if not imgs:
            print(f"[WARNING] No images found in directory '{path}'")
        return imgs
    else:
        print(f"[ERROR] Path not found: '{path}'")
        sys.exit(1)


def load_model(weights_path: Path) -> YOLO:
    """Load YOLO model from weights file with clear error messages."""
    if not weights_path.exists():
        print(f"\n[ERROR] Weights file not found: '{weights_path}'")
        print("        Check the path and try again.\n")
        sys.exit(1)

    print(f"  Loading model from: {weights_path}")
    try:
        model = YOLO(str(weights_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device            : {device.upper()}")
        if torch.cuda.is_available():
            print(f"  GPU               : {torch.cuda.get_device_name(0)}")
        return model
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}\n")
        sys.exit(1)


def draw_detections(image: np.ndarray, result) -> tuple[np.ndarray, dict]:
    """
    Draw bounding boxes and labels on image.
    Returns annotated image and detection summary dict.
    """
    annotated   = image.copy()
    counts      = {cls: 0 for cls in CLASS_NAMES}
    all_confs   = []

    if result.boxes is None or len(result.boxes) == 0:
        return annotated, counts, []

    boxes = result.boxes
    for box in boxes:
        cls_id  = int(box.cls[0])
        conf    = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        color    = CLASS_COLORS.get(cls_name, (200, 200, 200))

        counts[cls_name] = counts.get(cls_name, 0) + 1
        all_confs.append(conf)

        # Bounding box
        thickness = max(2, int(min(annotated.shape[:2]) / 300))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label background
        label       = f"{cls_name} {conf:.2f}"
        font_scale  = max(0.45, min(annotated.shape[1], annotated.shape[0]) / 1200)
        font_thick  = max(1, thickness - 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        label_y     = max(y1 - 6, th + 6)
        cv2.rectangle(annotated,
                      (x1, label_y - th - 6),
                      (x1 + tw + 6, label_y + 2),
                      color, -1)
        cv2.putText(annotated, label,
                    (x1 + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thick,
                    cv2.LINE_AA)

    return annotated, counts, all_confs


def print_detection_table(counts: dict, confs: list, elapsed_ms: float, img_name: str):
    """Print a clean detection summary table to the terminal."""
    total = sum(counts.values())
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Results for: {img_name[:27]:<27} │")
    print(f"  ├──────────────┬────────┬─────────────────┤")
    print(f"  │  Class       │ Count  │ Avg Confidence  │")
    print(f"  ├──────────────┼────────┼─────────────────┤")
    for cls in CLASS_NAMES:
        c = counts.get(cls, 0)
        print(f"  │  {cls:<12}│  {c:<5} │       —         │")
    print(f"  ├──────────────┼────────┼─────────────────┤")
    avg_conf = f"{sum(confs)/len(confs):.3f}" if confs else "—"
    print(f"  │  TOTAL       │  {total:<5} │  {avg_conf:<15} │")
    print(f"  ├──────────────┴────────┴─────────────────┤")
    print(f"  │  Inference time : {elapsed_ms:>6.1f} ms              │")
    fps = 1000 / elapsed_ms if elapsed_ms > 0 else 0
    print(f"  │  FPS            : {fps:>6.1f}                  │")
    print(f"  └─────────────────────────────────────────┘")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(BANNER)

    parser = build_parser()
    args   = parser.parse_args()

    weights_path = Path(args.weights)
    image_path   = Path(args.image)
    output_dir   = Path(args.output)

    # ── Load model ─────────────────────────────────────────
    print("[ Model ]")
    model  = load_model(weights_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[ Settings ]")
    print(f"  Confidence threshold : {args.conf}")
    print(f"  IoU threshold        : {args.iou}")
    print(f"  Image size           : {args.imgsz} × {args.imgsz}")

    # ── Collect images ─────────────────────────────────────
    images = find_images(image_path)
    if not images:
        print("[ERROR] No images to process.")
        sys.exit(1)

    print(f"\n[ Images ]")
    print(f"  Found {len(images)} image(s) to process")

    # ── Output dir ─────────────────────────────────────────
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[ Output ]")
        print(f"  Saving results to: {output_dir.resolve()}")

    # ── Run inference ──────────────────────────────────────
    print(f"\n[ Running Inference ]")
    print(f"  {'─'*55}")

    total_detections = 0
    total_time_ms    = 0.0
    batch_counts     = {cls: 0 for cls in CLASS_NAMES}

    for idx, img_path in enumerate(images, 1):
        print(f"\n  [{idx}/{len(images)}] {img_path.name}")

        # Read image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [WARNING] Could not read image: {img_path.name} — skipping")
            continue

        # Inference
        t0 = time.perf_counter()
        try:
            results = model.predict(
                source  = str(img_path),
                imgsz   = args.imgsz,
                conf    = args.conf,
                iou     = args.iou,
                device  = device,
                verbose = False,
            )
        except Exception as e:
            print(f"  [ERROR] Inference failed for {img_path.name}: {e}")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result    = results[0]
        annotated, counts, confs = draw_detections(img_bgr, result)

        # Update totals
        n_det = sum(counts.values())
        total_detections += n_det
        total_time_ms    += elapsed_ms
        for cls, cnt in counts.items():
            batch_counts[cls] += cnt

        # Print table
        print_detection_table(counts, confs, elapsed_ms, img_path.name)

        # Save result
        if not args.no_save:
            out_name = f"{img_path.stem}_detected{img_path.suffix}"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), annotated)
            print(f"  Saved → {out_path}")

        # Display
        if not args.no_show:
            display = cv2.resize(annotated, (
                min(annotated.shape[1], 1280),
                min(annotated.shape[0], 720),
            ))
            win_title = f"YOLOv10 — {img_path.name}  (press any key for next | Q to quit)"
            cv2.imshow(win_title, display)
            key = cv2.waitKey(0 if len(images) == 1 else 1500)
            cv2.destroyAllWindows()
            if key == ord("q") or key == ord("Q"):
                print("\n  [INFO] Quit by user.")
                break

    # ── Batch summary ──────────────────────────────────────
    if len(images) > 1:
        avg_time = total_time_ms / len(images) if images else 0
        avg_fps  = 1000 / avg_time if avg_time > 0 else 0
        print(f"\n{'─'*57}")
        print(f"  BATCH SUMMARY")
        print(f"{'─'*57}")
        print(f"  Images processed   : {len(images)}")
        print(f"  Total detections   : {total_detections}")
        for cls in CLASS_NAMES:
            print(f"    {cls:<14}  : {batch_counts[cls]}")
        print(f"  Avg inference time : {avg_time:.1f} ms/image")
        print(f"  Avg FPS            : {avg_fps:.1f}")
        if not args.no_save:
            print(f"  Results saved to   : {output_dir.resolve()}")
        print(f"{'─'*57}\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
