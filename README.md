# 🎓 EduVision 2026 — Classroom Crowd Detection & Student Counting

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Gradio-orange?style=for-the-badge)](https://6648f1f78220c70e48.gradio.live)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s_+_YOLOv8m-blue?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Trained_On-Kaggle_T4_GPU-20BEFF?style=for-the-badge)](https://kaggle.com)

> **🏆 Hackathon Submission — Team GSM | February 25, 2026**
>
> AI-powered system that detects and counts students in real-world classroom CCTV images using YOLOv8 with WBF ensemble and automated threshold tuning.

---

## 🎯 Live Demo

**👉 [Try it now: https://6648f1f78220c70e48.gradio.live](https://6648f1f78220c70e48.gradio.live)**

Upload any classroom image → Get instant bounding box detections + student count.

![demo](https://img.shields.io/badge/Status-LIVE-brightgreen?style=flat-square)

---

## 📸 Sample Results

| Input (Classroom CCTV) | Output (Detected Students) |
|:---:|:---:|
| Raw classroom image with 20+ students | Green bounding boxes with confidence scores |
| Varying angles, lighting, occlusions | Accurate count overlay on image |

---

## 🧠 Problem Statement

> **EduVision 2026**: Build a fully automated system that detects all people in classroom images (bounding boxes) and accurately estimates the student count per image.

### Evaluation Criteria
| Metric | Weight | Our Score |
|:---|:---:|:---:|
| **mAP@0.5** (Detection) | 50% | **0.9471** |
| **MAE** (Counting Error) | 50% | **0.3231** |
| Precision | — | 0.9256 |
| Recall | — | 0.9246 |

### Key Challenges
- 🏫 Dense classrooms with 5–50+ students per image
- 👥 Heavy occlusion (students overlapping behind desks)
- 📷 CCTV angles with perspective distortion
- 💡 Varying lighting conditions (daylight, fluorescent, shadows)
- 🪑 Furniture noise (chairs, desks, monitors)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EduVision 2026 Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CSV Annotations ──→ YOLO Format ──→ Train YOLOv8s (50ep)  │
│  (xmin,ymin,xmax,ymax)  (cx,cy,w,h)     ↓                  │
│                                     Train YOLOv8m (50ep)    │
│                                          ↓                  │
│                                   Threshold Tuning          │
│                                   (conf + IoU sweep)        │
│                                          ↓                  │
│                              ┌───────────┴──────────┐       │
│                              │   WBF Ensemble + TTA  │       │
│                              │  (2 models × 2 flips) │       │
│                              └───────────┬──────────┘       │
│                                          ↓                  │
│                                   Size Filtering            │
│                                          ↓                  │
│                              Bounding Boxes + Count         │
│                                          ↓                  │
│                              CSV + JSON + Gradio Demo       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Model Details

| Component | Details |
|:---|:---|
| **Primary Model** | YOLOv8s (11.2M params) — Fast, accurate, trains in ~45min on T4 |
| **Secondary Model** | YOLOv8m (25.9M params) — Higher capacity for ensemble boost |
| **Ensemble** | Weighted Boxes Fusion (WBF) with IoU=0.55 |
| **TTA** | Horizontal flip augmentation at inference |
| **Pretrained** | COCO weights (includes robust "person" detection) |
| **Fine-tuned On** | 4,236 classroom CCTV images |

---

## 📊 Dataset

| Property | Value |
|:---|:---|
| **Source** | EduVision 2026 Hackathon (Kaggle) |
| **Total Images** | 4,236 |
| **Train / Val Split** | 3,812 / 424 (90/10) |
| **Annotation Format** | CSV (`filename, width, height, class, xmin, ymin, xmax, ymax`) |
| **Classes** | `Person` (1 class) |
| **Avg People/Image** | ~25 |
| **Image Size** | 640×640 px |
| **Challenges** | Occlusion, CCTV angles, lighting variation, furniture noise |

---

## 🛠️ Training Configuration

| Parameter | YOLOv8s (Primary) | YOLOv8m (Ensemble) |
|:---|:---:|:---:|
| Epochs | 50 | 20 |
| Image Size | 640 | 640 |
| Batch Size | 16 | 8 |
| Optimizer | AdamW (lr=0.001) | SGD (lr=0.01) |
| Early Stopping | 15 epochs patience | 15 epochs patience |
| GPU | NVIDIA T4 | NVIDIA T4 |

### Augmentation Pipeline

| Augmentation | Value | Rationale |
|:---|:---:|:---|
| Mosaic | 1.0 | Multi-scale student detection |
| Mixup | 0.15 | Handle overlapping/occluded students |
| Copy-Paste | 0.1 | Increase instance density |
| HSV (H/S/V) | 0.015/0.7/0.4 | Lighting robustness |
| Scale | 0.5 | Different camera distances |
| Degrees | 10° | Tilted camera angles |
| Horizontal Flip | 0.5 | Directional invariance |

---

## 🎯 Post-Processing

1. **Confidence Threshold**: `0.35` (tuned on validation set to minimize MAE)
2. **NMS IoU Threshold**: `0.45` (prevents double-counting in dense seating)
3. **WBF Ensemble**: Merges 4 prediction sets (2 models × 2 TTA flips)
4. **Size Filtering**: Removes boxes < 3% image height (noise) or > 95% (background)
5. **Student Count**: Direct count of all filtered detections

---

## 📁 Project Structure

```
EduVision-2026/
├── README.md                          # This file
├── setup_and_eda.py                   # Step 1: Dataset exploration & analysis
├── convert_to_yolo.py                 # Step 2: CSV → YOLO format conversion
├── train_model.py                     # Step 3: Train YOLOv8s + YOLOv8m
├── evaluate_and_tune.py               # Step 4: Threshold tuning on validation
├── inference_and_submission.py         # Step 5: Generate predictions
├── visualize_results.py               # Step 6: Create visualizations
├── technical_report_generator.py      # Step 7: Auto-generate technical report
├── gradio_app.py                      # Step 8: Live Gradio demo
├── quick_demo.py                      # Fallback: Static demo image generator
├── prep_test_today.py                 # Pre-hackathon environment validator
│
├── submission/
│   ├── count_predictions.csv          # Per-image student count (for MAE)
│   ├── detection_predictions.json     # COCO-format bounding boxes (for mAP)
│   ├── technical_report.md            # Generated technical report
│   ├── demo/
│   │   └── demo_grid.png             # Static demo visualization
│   └── visualizations/
│       └── sample_predictions.png     # Prediction grid
│
└── runs/
    ├── yolov8s_eduvision/weights/best.pt   # Primary model weights
    └── yolov8m_eduvision/weights/best.pt   # Ensemble model weights
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
pip install ultralytics ensemble-boxes pycocotools albumentations gradio scikit-learn scipy supervision
```

### 2. Full Pipeline (Hackathon Execution Order)

```bash
# Step 1: Explore the dataset
python setup_and_eda.py

# Step 2: Convert CSV annotations to YOLO format
python convert_to_yolo.py

# Step 3: Train models (YOLOv8s primary + YOLOv8m if time permits)
python train_model.py

# Step 4: Tune confidence & IoU thresholds on validation set
python evaluate_and_tune.py

# Step 5: Run inference and generate submission files
python inference_and_submission.py

# Step 6: Create result visualizations
python visualize_results.py

# Step 7: Auto-generate technical report
python technical_report_generator.py

# Step 8: Launch live Gradio demo
python gradio_app.py
```

### 3. Inference Only (Using Pre-trained Weights)

```python
from ultralytics import YOLO

model = YOLO("runs/yolov8s_eduvision/weights/best.pt")
results = model("classroom_image.jpg", conf=0.35, iou=0.45)

# Count people
count = sum(len(r.boxes) for r in results)
print(f"Students detected: {count}")

# Visualize
for r in results:
    annotated = r.plot()
```

---

## 🔑 Key Design Decisions

### Why YOLOv8?
- **Speed**: YOLOv8s trains in ~45 min on T4 — critical for a 5-hour hackathon
- **Accuracy**: State-of-the-art mAP on person detection
- **COCO Pretrained**: Already understands "person" class from 330K images
- **Ecosystem**: Stable NMS, easy WBF integration, Gradio-compatible

### Why Ensemble?
- WBF fusion of YOLOv8s + YOLOv8m reduces false positives by cross-validating predictions
- TTA (horizontal flip) catches students missed due to asymmetric occlusion
- Net result: **Lower MAE** with minimal computational overhead

### Safety-First Architecture
- YOLOv8s trained **first** → guarantees a submission even if time runs out
- **Time-Check algorithm** auto-decides whether to train YOLOv8m based on remaining hackathon minutes
- Config chain (JSON files) automatically passes tuned parameters between scripts — zero manual copying

### Automated Threshold Tuning
- Exhaustive grid search over `conf ∈ [0.10, 0.50]` and `iou ∈ [0.30, 0.60]`
- Optimized directly for **MAE** (the competition counting metric)
- Best found: `conf=0.35, iou=0.45`

---

## 📈 Results Summary

```
┌────────────────────────┬──────────┐
│ Metric                 │ Score    │
├────────────────────────┼──────────┤
│ mAP@0.5               │ 0.9471   │
│ mAP@0.5:0.95          │ 0.7743   │
│ Precision              │ 0.9256   │
│ Recall                 │ 0.9246   │
│ Counting MAE           │ 0.3231   │
│ Avg Inference Time     │ ~15ms    │
└────────────────────────┴──────────┘
```

---

## 🌐 Live Demo

**🔗 [https://6648f1f78220c70e48.gradio.live](https://6648f1f78220c70e48.gradio.live)**

Features:
- 📷 Upload any classroom image
- 🟢 Green bounding boxes around detected people
- 👥 Real-time student count
- 🎚️ Adjustable confidence & IoU sliders
- 📊 Detailed detection report per image
- 📁 Pre-loaded example images from the dataset

---

## 👥 Team

**Team GSM** — EduVision 2026 Hackathon

---

## 📜 License

This project was built for the **EduVision 2026 Hackathon**. All code is provided for educational and competition purposes.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection framework
- [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) — Ensemble method
- [Gradio](https://gradio.app/) — Live demo UI
- [Kaggle](https://kaggle.com) — GPU compute & dataset hosting
- EduVision 2026 organizers for the dataset and problem statement

---

<p align="center">
  <b>Built with ❤️ during EduVision 2026 Hackathon</b><br>
  <i>From raw CCTV images to real-time student counting in 5 hours</i>
</p>
