# EduVision 2026 — Technical Report
**Team:** GSM
**Date:** 2026-02-25
**Inference Mode:** YOLOv8s + YOLOv8m WBF Ensemble

---

## 1. Problem Approach

### Overall Strategy
We implemented a robust object detection pipeline using the YOLOv8 architecture. Given the dense nature of the classroom dataset (avg. 25 students per image), we prioritized models that handle occlusion and small-object detection effectively. The pipeline was designed to be "Time-Aware," ensuring a high-quality submission within the 5-hour hackathon window.

### Architecture
* **Primary Model**: **YOLOv8s (Small)** — Selected for its optimal balance of speed and high mAP on the Kaggle T4 GPU.
* **Secondary Model**: **YOLOv8m (Medium)** — Trained as a complementary model to boost accuracy through ensembling.
* **Ensemble Method**: **Weighted Boxes Fusion (WBF)** with Test-Time Augmentation (TTA) — Merges predictions from both models + horizontal flips to reduce False Positives and improve MAE.
* **Classes**: `person (1-class)`

---

## 2. Data Preprocessing & Augmentation

### Annotation Format
* **Source Format**: CSV with columns `filename, width, height, class, xmin, ymin, xmax, ymax`
* **Conversion**: Absolute pixel coordinates (xyxy) to Normalized YOLO format (cx, cy, w, h).
* **Auto-Detection**: Dynamic path-finder to locate `_annotations.csv` across the Kaggle environment.
* **Dataset Size**: 4,236 images (3,812 Train / 424 Val).
* **Validation**: Coordinate clamping to [0.001, 0.999] and tiny box filtering (< 3px).

### Augmentation Pipeline (Optimized for Dense Classrooms)
| Augmentation | Value | Rationale |
| :--- | :--- | :--- |
| **Mosaic** | 1.0 | Forces model to detect students at multiple scales |
| **Mixup** | 0.15 | Helps distinguish overlapping/occluded students |
| **Copy-Paste** | 0.1 | Increases instance density for small-object detection |
| **HSV (H/S/V)** | 0.015 / 0.7 / 0.4 | Robustness against varying classroom lighting |
| **Scale** | 0.5 | Handle different camera distances |
| **Degrees** | 10 deg | Handle tilted camera angles |

---

## 3. Training Configuration

| Parameter | Model 1 (YOLOv8s) | Model 2 (YOLOv8m) |
| :--- | :--- | :--- |
| **Pretrained** | COCO | COCO |
| **Epochs** | 50 | 20 |
| **Image Size** | 640 | 640 |
| **Batch Size** | 16 | 8 |
| **Optimizer** | AdamW (lr=0.001) | SGD (lr=0.01) |
| **GPU** | NVIDIA T4 | NVIDIA T4 |

---

## 4. Post-Processing & Counting

1. **Confidence Threshold**: **0.30 - 0.35** (Tuned to minimize MAE).
2. **NMS IoU Threshold**: **0.45** (Prevents double-counting in dense seating).
3. **Size Filtering**: Boxes < 3% image height (noise) or > 95% (background) removed.
4. **WBF Ensemble**: IoU=0.55 across 4 prediction sets (2 models x 2 TTA flips).
5. **Student Count**: Direct count of all filtered detections.

---

## 5. Results (Verified on Validation Set)

### Validation Metrics
| Metric | Score |
| :--- | :--- |
| **mAP@0.5** | **0.9471** |
| **mAP@0.5:0.95** | **0.7743** |
| **Precision** | **0.9256** |
| **Recall** | **0.9246** |
| **Counting MAE** | **0.3231** |

---

## 6. Key Design Decisions

### Strategic "Safety First" Training
We trained YOLOv8s first to guarantee a valid, high-quality submission early. A **Time-Check algorithm** dynamically decided whether to train the second model based on the remaining hackathon window.

### Why YOLOv8 over alternatives?
* **YOLOv8s** (11.2M params) trains in ~45 min on T4 — ideal for strict deadlines.
* **Stability**: Mature ecosystem with stable NMS and WBF ensembling.
* **Accuracy**: Excellent performance on "person" detection out-of-the-box.

---

## 7. Deliverables
* `count_predictions.csv`: Final leaderboard counts.
* `detection_predictions.json`: COCO-format bounding box predictions.
* `best.pt`: Model weights (v8s + v8m).
* **Live Demo**: Gradio web application for real-time visualization.


---

## 8. Reproduction Instructions

```bash
# 1. Environment Setup
pip install ultralytics ensemble-boxes pycocotools albumentations gradio scikit-learn

# 2. Data Preparation
python setup_and_eda.py              # EDA + environment check
python convert_to_yolo.py            # CSV to YOLO format conversion

# 3. Training
python train_model.py                # Train YOLOv8s (+ YOLOv8m if time permits)

# 4. Evaluation & Inference
python evaluate_and_tune.py          # Tune confidence/IoU thresholds
python inference_and_submission.py   # Generate final predictions

# 5. Visualization & Demo
python visualize_results.py          # Create visual outputs
python gradio_app.py                 # Launch live demo
```

---

*Generated automatically by EduVision 2026 pipeline.*
