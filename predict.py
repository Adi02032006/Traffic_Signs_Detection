"""
YOLOv11 Indian Traffic Sign Detection — Inference Script
=========================================================
Runs inference on images or video to detect and classify traffic signs.

Usage:
    python predict.py --source image.jpg
    python predict.py --source video.mp4
    python predict.py --source path/to/folder/
    python predict.py --source 0                  # webcam
"""

from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Indian Traffic Sign Detection — Inference")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image, video, folder, or '0' for webcam")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (default: best.pt or yolo11s.pt)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save results (default: True)")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Show results in window")
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    print("YOLOv11 Indian Traffic Sign — Inference")
    print("=" * 60)
    print(f"  Model:       {model_weights}")
    print(f"  Source:      {args.source}")
    print(f"  Confidence:  {args.conf}")
    print(f"  IoU:         {args.iou}")
    print()
    
    # Load model
    model = YOLO(model_weights)
    
    # Run inference
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=640,
        device=0,
        save=args.save,
        show=args.show,
        save_txt=True,          # Save detection results as .txt
        save_conf=True,         # Include confidence in saved txt
        line_width=2,
        show_labels=True,
        show_conf=True,
        project=os.path.join(BASE_DIR, "runs", "predict"),
        name="results",
        exist_ok=True,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    
    total_detections = 0
    for r in results:
        n = len(r.boxes)
        total_detections += n
        if n > 0:
            print(f"\n  {r.path}:")
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"    [{conf:.2f}] {cls_name} @ ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
    
    print(f"\n  Total detections: {total_detections}")
    if args.save:
        print(f"  Results saved to: runs/predict/results/")


if __name__ == "__main__":
    main()
