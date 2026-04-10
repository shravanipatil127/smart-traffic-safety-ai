"""
train.py — Fine-tune YOLOv8 on your custom helmet/seatbelt dataset

BEFORE RUNNING:
  1. Download your dataset from Roboflow (export as YOLOv8 format)
  2. Place it in the dataset/ folder
  3. Update DATA_YAML path below
  4. Run: python train.py

Dataset structure expected:
  dataset/
    data.yaml          ← class names + train/val paths
    train/
      images/
      labels/
    valid/
      images/
      labels/
"""

from ultralytics import YOLO
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────

DATA_YAML   = "dataset/data.yaml"   # Path to your Roboflow data.yaml
MODEL_BASE  = "yolov8n.pt"          # Base model: yolov8n / yolov8s / yolov8m
EPOCHS      = 50                    # Increase to 100 for better accuracy
IMG_SIZE    = 640
BATCH_SIZE  = 16                    # Reduce to 8 if GPU runs out of memory
OUTPUT_DIR  = "models"

# ────────────────────────────────────────────────────────────────────────────


def train():
    if not os.path.exists(DATA_YAML):
        print(f"[ERROR] data.yaml not found at: {DATA_YAML}")
        print("  → Download your dataset from Roboflow and place it in dataset/")
        print("  → Export format: YOLOv8")
        return

    print(f"[TRAIN] Starting fine-tune on {MODEL_BASE} for {EPOCHS} epochs...")

    model = YOLO(MODEL_BASE)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=OUTPUT_DIR,
        name="helmet_seatbelt_v1",
        patience=15,            # Early stopping
        save=True,
        save_period=10,         # Save checkpoint every N epochs
        val=True,
        plots=True,             # Saves training curves in output folder
        workers=4,
        device=0,               # GPU index; use 'cpu' if no GPU
        verbose=True,
    )

    best_weights = os.path.join(OUTPUT_DIR, "helmet_seatbelt_v1", "weights", "best.pt")
    print(f"\n[DONE] Training complete!")
    print(f"  Best weights saved at: {best_weights}")
    print(f"  Copy to models/best.pt and run detect.py")


def evaluate():
    """Run validation metrics on the trained model."""
    weights = os.path.join(OUTPUT_DIR, "helmet_seatbelt_v1", "weights", "best.pt")
    if not os.path.exists(weights):
        print("[ERROR] No trained model found. Run train() first.")
        return

    model = YOLO(weights)
    metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE)

    print(f"\n[METRICS]")
    print(f"  mAP@0.5     : {metrics.box.map50:.3f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"  Precision   : {metrics.box.p.mean():.3f}")
    print(f"  Recall      : {metrics.box.r.mean():.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of training")
    args = parser.parse_args()

    if args.eval:
        evaluate()
    else:
        train()
