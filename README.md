# Indian Traffic Sign Detection and Classification with YOLO11

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg?style=flat)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](LICENSE)
[![mAP50](https://img.shields.io/badge/mAP%4050-91.28%25-success.svg?style=flat)](runs/detect/indian_traffic_signs_merged/results.png)

A deep learning framework for real-time detection and classification of **104 distinct classes of Indian traffic and road signs** using **Ultralytics YOLO11s**. Designed for Intelligent Transportation Systems (ITS), Advanced Driver Assistance Systems (ADAS), and autonomous navigation under diverse Indian road conditions.

---

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Performance Benchmarks](#performance-benchmarks)
- [Training Visualizations and Metrics](#training-visualizations-and-metrics)
- [Dataset Composition](#dataset-composition)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Inference and Usage](#inference-and-usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [Evaluation](#evaluation)
- [Training Procedure](#training-procedure)
- [Supported Traffic Sign Classes](#supported-traffic-sign-classes)
- [References and Citations](#references-and-citations)
- [License](#license)

---

## Introduction

Road sign recognition in the Indian traffic ecosystem presents unique challenges due to complex background clutter, dense traffic flow, high variation in illumination, weathered signboards, and regional signage variations. 

This repository provides an end-to-end pipeline covering dataset configuration, fine-tuning, evaluation, and production-ready inference using the YOLO11 small (`yolo11s`) architecture. It addresses both standard regulatory signs and localized signboards across 104 distinct categories.

---

## Key Features

- **Optimized Model Architecture**: Employs YOLO11s for an optimal trade-off between floating-point operations (FLOPs), parameter efficiency, and mean average precision.
- **Comprehensive Class Vocabulary**: Supports 104 categories encompassing mandatory (regulatory), cautionary (warning), and informatory traffic signs.
- **Standardized Multi-Source Dataset**: Curated and merged from prominent traffic sign benchmarks (Roboflow Universe, Kaggle, IRTSD, INTSD).
- **Data Augmentation Strategy**: Uses Mosaic, MixUp, Copy-Paste, HSV color shifts, and non-destructive spatial transformations tailored to preserve sign orientation and geometry.
- **Flexible Inference Modes**: Built-in support for single images, video sequences, directory batches, and live camera streams.

---

## Performance Benchmarks

The model was evaluated on held-out test splits. Key performance indicators are summarized below:

| Metric | Value |
| :--- | :--- |
| **mAP@50 (Mean Average Precision @ IoU 0.50)** | **91.28%** |
| **mAP@50-95 (Mean Average Precision @ IoU 0.50:0.95)** | **79.68%** |
| **Precision (Mean Precision across classes)** | **84.04%** |
| **Recall (Mean Recall across classes)** | **90.28%** |
| **Input Image Resolution** | `640 x 640` |
| **Model Size** | Small (`yolo11s`) |
| **Batch Size** | 16 |

---

## Training Visualizations and Metrics

### Loss and Metric Convergence
The training progression across epochs illustrating bounding box loss, classification loss, distribution focal loss (DFL), precision, recall, and mAP metrics:

![Training Metrics](runs/detect/indian_traffic_signs_merged/results.png)

### Normalized Confusion Matrix
Detailed per-class prediction accuracy across categories:

![Normalized Confusion Matrix](runs/detect/indian_traffic_signs_merged/confusion_matrix_normalized.png)

### Diagnostic Curves
- **Precision-Recall Curve**: [BoxPR_curve.png](runs/detect/indian_traffic_signs_merged/BoxPR_curve.png)
- **F1-Confidence Curve**: [BoxF1_curve.png](runs/detect/indian_traffic_signs_merged/BoxF1_curve.png)

### Sample Detections
Validation predictions demonstrating accurate bounding box localization and classification in multi-sign scenarios:

![Validation Batch Predictions](runs/detect/indian_traffic_signs_merged/val_batch0_pred.jpg)

---

## Dataset Composition

The training data aggregates multiple public and research datasets standardized into YOLO format:
- **Roboflow Universe**: Indian Traffic Sign benchmarks ([Link 1](https://universe.roboflow.com/sdi/indian-traffic-sign), [Link 2](https://universe.roboflow.com/college-opfvn/india-traffic-sign), [Link 3](https://universe.roboflow.com/major-project-166na/indian-traffic-signboards))
- **Kaggle**: Indian Traffic Sign Dataset ([Link](https://www.kaggle.com/datasets/neelpratiksha/indian-traffic-sign-dataset))
- **IEEE DataPort**: IRTSD Dataset v1 - Indian Road Traffic Sign Detection ([Link](https://ieee-dataport.org/documents/irtsd-datasetv1-indian-road-traffic-sign-detection-dataset))
- **INTSD Benchmark**: Indian Traffic Sign Dataset ([Link](https://adityamishra-ml.github.io/INTSD/download.html))
- Additional references and source links are cataloged in [`dataset.txt`](dataset.txt).

The dataset configuration file containing paths and category labels is maintained in [`merged_dataset/data.yaml`](merged_dataset/data.yaml).

---

## Repository Structure

```
Traffic_Signs_Detection/
|-- merged_dataset/
|   `-- data.yaml                  # YOLO dataset configuration and class mapping
|-- runs/
|   `-- detect/
|       `-- indian_traffic_signs_merged/
|           |-- args.yaml          # Hyperparameters and training arguments
|           |-- results.csv        # Tabulated epoch-by-epoch training logs
|           |-- results.png        # Training loss and evaluation metric plots
|           |-- confusion_matrix_normalized.png
|           |-- BoxF1_curve.png
|           |-- BoxPR_curve.png
|           `-- val_batch0_pred.jpg
|-- train.py                       # Model training script
|-- predict.py                     # Inference script (Images, Video, Webcam)
|-- evaluate.py                    # Evaluation script for test set validation
|-- requirements.txt               # Project dependencies
|-- dataset.txt                    # Dataset citations and source links
|-- yolo11s.pt                     # YOLO11 Small model checkpoint
|-- .gitignore                     # Git ignore configuration
`-- README.md                      # Technical project documentation
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Adi02032006/Traffic_Signs_Detection.git
cd Traffic_Signs_Detection
```

### 2. Set Up a Virtual Environment
```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Inference and Usage

### Command Line Interface

Run inference on single images, videos, directories, or a live webcam feed using [`predict.py`](predict.py):

```bash
# Inference on a single image
python predict.py --source path/to/image.jpg

# Inference on a video file
python predict.py --source path/to/video.mp4

# Real-time inference using a webcam (device index 0)
python predict.py --source 0

# Adjust confidence and NMS IoU thresholds
python predict.py --source path/to/image.jpg --conf 0.40 --iou 0.50

# Specify explicit weights file
python predict.py --source path/to/image.jpg --weights yolo11s.pt
```

### Python API

Integrate the model directly into custom Python pipelines:

```python
from ultralytics import YOLO

# Load model checkpoint
model = YOLO("yolo11s.pt")

# Execute inference
results = model.predict(
    source="path/to/image.jpg",
    conf=0.35,
    iou=0.45,
    save=True
)

# Extract and process detections
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()
        print(f"Class: {class_name} | Confidence: {confidence:.2f} | Bounding Box: {coordinates}")
```

---

## Evaluation

To evaluate model checkpoints on the test partition:

```bash
# Evaluate using default parameters
python evaluate.py

# Evaluate with custom weights and configuration
python evaluate.py --weights yolo11s.pt --data merged_dataset/data.yaml --split test
```

The script prints aggregated summary statistics (mAP@50, mAP@50-95, Precision, Recall) followed by per-class average precision rankings.

---

## Training Procedure

To train or fine-tune the model on your hardware:

```bash
python train.py
```

### Key Hyperparameters
- **Base Architecture**: `yolo11s.pt`
- **Optimizer**: AdamW (`lr0=0.001`, `lrf=0.01`, Cosine LR scheduling)
- **Input Resolution**: `640 x 640`
- **Batch Size**: 16 (configured for 8 GB VRAM)
- **Data Augmentations**: Mosaic (1.0), MixUp (0.1), Copy-Paste (0.1), Random Erasing (0.1)
- **Early Stopping**: Patience of 15 epochs

---

## Supported Traffic Sign Classes

The model detects and classifies 104 categories of Indian traffic signs:

<details>
<summary>Click to expand full class index (0 to 103)</summary>

| ID | Class Name | ID | Class Name |
| :--- | :--- | :--- | :--- |
| 0 | ACCIDENT | 52 | NO_ENTRY |
| 1 | AIRPORT | 53 | NO_PARKING |
| 2 | ALL_MOTOR_VEHICLE_PROHIBITED | 54 | NO_STOPPING_OR_STANDING |
| 3 | AXLE_LOAD_LIMIT | 55 | OVERTAKING_PROHIBITED |
| 4 | BARRIER_AHEAD | 56 | PARKING |
| 5 | BULLOCK_AND_HANDCART_PROHIBITED | 57 | PASS_EITHER_SIDE |
| 6 | BULLOCK_CART_AND_HAND_CART_PROHIBITED | 58 | PEDESTRIAN_CROSSING |
| 7 | BULLOCK_PROHIBITED | 59 | PEDESTRIAN_PROHIBITED |
| 8 | BUS_STOP | 60 | PETROL_PUMP_AHEAD |
| 9 | CATTLE | 61 | PRIORITY_FOR_ONCOMING_VEHICLES |
| 10 | CHEVRON_DIRECTION | 62 | QUAY_SIDE_OR_RIVER_BANK |
| 11 | COMPULSARY_AHEAD | 63 | RESTRICTION_ENDS |
| 12 | COMPULSARY_AHEAD_OR_TURN_LEFT | 64 | RIGHT_HAIR_PIN_BEND |
| 13 | COMPULSARY_AHEAD_OR_TURN_RIGHT | 65 | RIGHT_HAND_CURVE |
| 14 | COMPULSARY_CYCLE_TRACK | 66 | RIGHT_REVERSE_BEND |
| 15 | COMPULSARY_KEEP_LEFT | 67 | RIGHT_TURN_PROHIBITED |
| 16 | COMPULSARY_KEEP_RIGHT | 68 | ROAD_WIDENS_AHEAD |
| 17 | COMPULSARY_MINIMUM_SPEED | 69 | ROUNDABOUT |
| 18 | COMPULSARY_SOUND_HORN | 70 | SCHOOL_AHEAD |
| 19 | COMPULSARY_TURN_LEFT | 71 | SIDE_ROAD_LEFT |
| 20 | COMPULSARY_TURN_LEFT_AHEAD | 72 | SIDE_ROAD_RIGHT |
| 21 | COMPULSARY_TURN_RIGHT | 73 | SLIPPERY_ROAD |
| 22 | COMPULSARY_TURN_RIGHT_AHEAD | 74 | SPEED_BREAKER |
| 23 | COMPULSORY_LEFT | 75 | SPEED_LIMIT |
| 24 | COMPULSORY_RIGHT | 76 | SPEED_LIMIT_100 |
| 25 | CROSS_ROAD | 77 | SPEED_LIMIT_120 |
| 26 | CYCLE_CROSSING | 78 | SPEED_LIMIT_15 |
| 27 | CYCLE_PROHIBITED | 79 | SPEED_LIMIT_20 |
| 28 | DANGEROUS_DIP | 80 | SPEED_LIMIT_30 |
| 29 | FALLING_ROCKS | 81 | SPEED_LIMIT_40 |
| 30 | FERRY | 82 | SPEED_LIMIT_5 |
| 31 | GAP_IN_MEDIAN | 83 | SPEED_LIMIT_50 |
| 32 | GIVE_WAY | 84 | SPEED_LIMIT_60 |
| 33 | GO_SLOW | 85 | SPEED_LIMIT_70 |
| 34 | GUARDED_LEVEL_CROSSING | 86 | SPEED_LIMIT_80 |
| 35 | HANDCART_PROHIBITED | 87 | STAGGERED_INTERSECTION |
| 36 | HEIGHT_LIMIT | 88 | STEEP_ASCENT |
| 37 | HORN_PROHIBITED | 89 | STEEP_DESCENT |
| 38 | HORSE_CAR_PROHIBITED | 90 | STOP |
| 39 | HOSPITAL_AHEAD | 91 | STRAIGHT_PROHIBITED |
| 40 | HUMP_OR_ROUGH_ROAD | 92 | TONGA_PROHIBITED |
| 41 | LEFT_HAIR_PIN_BEND | 93 | TRAFFIC_SIGNAL |
| 42 | LEFT_HAND_CURVE | 94 | TRUCK_PROHIBITED |
| 43 | LEFT_REVERSE_BEND | 95 | TURN_RIGHT |
| 44 | LEFT_TURN_PROHIBITED | 96 | TWO_WAY_SIGNS |
| 45 | LENGTH_LIMIT | 97 | T_INTERSECTION |
| 46 | LOAD_LIMIT | 98 | UNGUARDED_LEVEL_CROSSING |
| 47 | LOOSE_GRAVEL | 99 | U_TURN |
| 48 | MEN_AT_WORK | 100 | U_TURN_PROHIBITED |
| 49 | NARROW_BRIDGE_AHEAD | 101 | WARNING |
| 50 | NARROW_ROAD_AHEAD | 102 | WIDTH_LIMIT |
| 51 | NARROW_ROAD_LEFT_AHEAD | 103 | Y_INTERSECTION |

</details>

---

## References and Citations

- **Ultralytics YOLO11**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Indian Traffic Sign Dataset (INTSD)**: [ArXiv Preprint 2511.17183](https://arxiv.org/pdf/2511.17183)
- **Indian Road Traffic Sign Detection (IRTSD)**: [IEEE DataPort Benchmark](https://ieee-dataport.org/documents/irtsd-datasetv1-indian-road-traffic-sign-detection-dataset)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
