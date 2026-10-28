"""
YOLOv11 Indian Traffic Sign Detection — Training Script
========================================================
Trains a YOLOv11s model on the 85-class Indian Traffic Sign dataset.
Optimized for RTX 4060 (8GB VRAM).

Usage:
    python train.py
"""

from ultralytics import YOLO
import os
import torch

def main():
    # ──────────────────────────────────────────────
    # Configuration
    # ──────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_YAML = os.path.join(BASE_DIR, "merged_dataset", "data.yaml")
    
    MODEL_SIZE = "yolo11s.pt"    # s = small, good balance of speed & accuracy
    EPOCHS = 100                  # 100 epochs with early stopping (patience=15)
    IMG_SIZE = 640                # Standard YOLO input size
    BATCH_SIZE = 16               # RTX 4060 8GB VRAM — safe batch size
    WORKERS = 8                   # Dataloader workers
    PROJECT = os.path.join(BASE_DIR, "runs", "detect")
    NAME = "indian_traffic_signs_merged"

    # ──────────────────────────────────────────────
    # Sanity checks
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("YOLOv11 Indian Traffic Sign Training")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        print("No GPU detected — training will be very slow!")
        BATCH_SIZE = 4

    # Check dataset
    if not os.path.exists(DATA_YAML):
        print(f"Dataset not found at: {DATA_YAML}")
        print("  Run the extraction script first!")
        return
    print(f"Dataset: {DATA_YAML}")

    # ──────────────────────────────────────────────
    # Load model
    # ──────────────────────────────────────────────
    print(f"\nLoading pretrained {MODEL_SIZE}...")
    model = YOLO(MODEL_SIZE)

    # ──────────────────────────────────────────────
    # Train
    # ──────────────────────────────────────────────
    print(f"\nStarting training:")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Workers:    {WORKERS}")
    print(f"  Output:     {PROJECT}/{NAME}")
    print()

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        
        # ── Optimizer ──
        optimizer="AdamW",
        lr0=0.001,              # Initial learning rate
        lrf=0.01,               # Final LR = lr0 * lrf
        cos_lr=True,            # Cosine annealing schedule
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,

        # ── Augmentations ──
        hsv_h=0.015,            # HSV-Hue augmentation
        hsv_s=0.7,              # HSV-Saturation
        hsv_v=0.4,              # HSV-Value
        degrees=10.0,           # Rotation (+/- deg)
        translate=0.1,          # Translation
        scale=0.5,              # Scale
        shear=2.0,              # Shear
        flipud=0.0,             # No vertical flip (signs have orientation)
        fliplr=0.0,             # No horizontal flip (signs like left/right curve matter)
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.1,              # Mixup augmentation
        copy_paste=0.1,         # Copy-paste augmentation (helps rare classes)
        erasing=0.1,            # Random erasing

        # ── Training strategy ──
        patience=15,            # Early stopping patience
        save=True,
        save_period=10,         # Save checkpoint every 10 epochs
        val=True,
        plots=True,             # Generate training plots
        
        # ── Performance ──
        amp=True,               # Mixed precision (faster on RTX 4060)
        cache=False,            # Don't cache to RAM (14K images is too many)
        device=0,               # GPU 0
        
        # ── Label smoothing ──
        label_smoothing=0.05,   # Slight smoothing to help with noisy labels
        
        exist_ok=True,          # Overwrite existing run
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best weights: {PROJECT}/{NAME}/weights/best.pt")
    print(f"Last weights: {PROJECT}/{NAME}/weights/last.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
