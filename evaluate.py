"""
YOLOv11 Indian Traffic Sign Detection — Evaluation Script
==========================================================
Evaluates the trained model on the test set and generates metrics.

Usage:
    python evaluate.py
"""

import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Indian Traffic Sign Detection — Evaluation")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (default: best.pt or yolo11s.pt)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data.yaml (default: merged_dataset/data.yaml)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate: 'test' or 'val' (default: test)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_YAML = args.data or os.path.join(BASE_DIR, "merged_dataset", "data.yaml")
    
    # Find weights
    model_weights = args.weights
    if not model_weights:
        candidates = [
            os.path.join(BASE_DIR, "runs", "detect", "indian_traffic_signs_merged", "weights", "best.pt"),
            os.path.join(BASE_DIR, "yolo11s.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_weights = c
                break
    
    if not model_weights or not os.path.exists(model_weights):
        print(f"Model weights not found! Please provide --weights path.")
        print("Train the model first with: python train.py")
        return
    
    print("=" * 60)
    print("YOLOv11 Indian Traffic Sign — Evaluation")
    print("=" * 60)
    print(f"Model: {model_weights}")
    print(f"Data:  {DATA_YAML}")
    print()
    
    # Load model
    model = YOLO(model_weights)
    
    # ── Validate on test set ──
    print("Running evaluation on test set...")
    metrics = model.val(
        data=DATA_YAML,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        plots=True,
        save_json=True,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")
    print()
    
    # Per-class results
    print("Per-class AP@50 (top 20):")
    print("-" * 40)
    
    class_names = model.names
    ap50_per_class = metrics.box.ap50
    
    # Sort by AP50 descending
    class_results = []
    for i, ap in enumerate(ap50_per_class):
        name = class_names.get(i, f"class_{i}")
        class_results.append((name, ap))
    
    class_results.sort(key=lambda x: x[1], reverse=True)
    
    for name, ap in class_results[:20]:
        bar = "#" * int(ap * 30)
        print(f"  {name:>40}: {ap:.3f} |{bar}")
    
    if len(class_results) > 20:
        print(f"\n  ... and {len(class_results) - 20} more classes")
    
    # Worst performing classes
    print(f"\nWorst 10 classes:")
    print("-" * 40)
    for name, ap in class_results[-10:]:
        bar = "#" * int(ap * 30)
        print(f"  {name:>40}: {ap:.3f} |{bar}")
    
    print(f"\nDetailed results saved to: runs/detect/")


if __name__ == "__main__":
    main()
