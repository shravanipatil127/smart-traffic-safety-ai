"""
Helmet & Seatbelt Detection System
Author: Your Name
Description: Real-time detection of helmet and seatbelt violations using YOLOv8
"""

import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path
from ultralytics import YOLO
from utils.draw import draw_detections, draw_stats
from utils.logger import ViolationLogger


def run_detection(source=0, model_path="models/best.pt", conf=0.5, save_output=False):
    """
    Run helmet/seatbelt detection on video source.

    Args:
        source: 0 for webcam, or path to video/image file
        model_path: Path to trained YOLOv8 model weights
        conf: Confidence threshold (0.0 - 1.0)
        save_output: Whether to save annotated output video
    """

    # Load model (falls back to pretrained YOLOv8n if custom weights not found)
    if not os.path.exists(model_path):
        print(f"[INFO] Custom model not found at {model_path}. Using base YOLOv8n.")
        model = YOLO("yolov8n.pt")
    else:
        print(f"[INFO] Loading model from {model_path}")
        model = YOLO(model_path)

    # Class names — update these to match your trained model's classes
    CLASS_NAMES = {
        0: "helmet",
        1: "no_helmet",
        2: "seatbelt",
        3: "no_seatbelt",
    }

    # Colors: green = safe, red = violation
    CLASS_COLORS = {
        "helmet":     (0, 200, 80),
        "no_helmet":  (0, 50, 220),
        "seatbelt":   (0, 200, 80),
        "no_seatbelt":(0, 50, 220),
    }

    # Open video source
    is_image = isinstance(source, str) and source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Cannot read image: {source}")
            return
        frames = [frame]
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {source}")
            return

    # Video writer setup
    writer = None
    if save_output and not is_image:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        os.makedirs("results", exist_ok=True)
        out_path = f"results/output_{int(time.time())}.avi"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
        print(f"[INFO] Saving output to {out_path}")

    logger = ViolationLogger()
    frame_count = 0
    violation_count = 0

    print("[INFO] Starting detection. Press 'q' to quit.")

    while True:
        if is_image:
            if frame_count >= len(frames):
                break
            frame = frames[frame_count]
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame_count += 1
        t_start = time.time()

        # Run YOLOv8 inference
        results = model(frame, conf=conf, verbose=False)[0]

        # Parse detections
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Map class id to name (handles custom or COCO classes)
            if cls_id in CLASS_NAMES:
                label = CLASS_NAMES[cls_id]
            else:
                label = model.names.get(cls_id, str(cls_id))

            is_violation = "no_" in label
            if is_violation:
                violation_count += 1
                logger.log(label, conf_score, frame_count)

            detections.append({
                "box": (x1, y1, x2, y2),
                "label": label,
                "conf": conf_score,
                "color": CLASS_COLORS.get(label, (200, 200, 200)),
                "violation": is_violation,
            })

        fps_display = 1.0 / max(time.time() - t_start, 1e-6)

        # Draw results on frame
        annotated = draw_detections(frame.copy(), detections)
        annotated = draw_stats(annotated, frame_count, violation_count, fps_display)

        if save_output and writer:
            writer.write(annotated)

        if is_image:
            # For images: save and show
            out_img = f"results/detected_{Path(source).stem}.jpg"
            os.makedirs("results", exist_ok=True)
            cv2.imwrite(out_img, annotated)
            print(f"[INFO] Saved result to {out_img}")
            cv2.imshow("Helmet & Seatbelt Detection", annotated)
            cv2.waitKey(0)
            break
        else:
            cv2.imshow("Helmet & Seatbelt Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Cleanup
    if not is_image:
        cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n[SUMMARY]")
    print(f"  Frames processed : {frame_count}")
    print(f"  Violations found : {violation_count}")
    logger.save_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet & Seatbelt Violation Detector")
    parser.add_argument("--source", default=0,
                        help="Video source: 0=webcam, or path to video/image file")
    parser.add_argument("--model", default="models/best.pt",
                        help="Path to YOLOv8 model weights")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Detection confidence threshold (0.0–1.0)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output video to results/")
    args = parser.parse_args()

    # Auto-convert source to int if it's a digit string (webcam index)
    source = int(args.source) if str(args.source).isdigit() else args.source
    run_detection(source=source, model_path=args.model, conf=args.conf, save_output=args.save)
