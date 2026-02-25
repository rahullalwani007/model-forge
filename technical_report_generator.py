# ============================================================
# EduVision 2026 - Step 7: Auto-generate Technical Report
# Generates the exact report format, saves to disk, and prints
# ============================================================

import os
import json
import pandas as pd
from datetime import datetime

OUTPUT_DIR = "/kaggle/working/submission"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. SAFE LOAD ALL AVAILABLE DATA
# ============================================================

# Safe load count predictions
count_path = f"{OUTPUT_DIR}/count_predictions.csv"
df_sub = pd.read_csv(count_path) if os.path.exists(count_path) else None

# Safe load detection predictions
det_path = f"{OUTPUT_DIR}/detection_predictions.json"
if os.path.exists(det_path):
    with open(det_path) as f:
        det_preds = json.load(f)
else:
    det_preds = None

# Load class config
config_path = "/kaggle/working/class_config.json"
class_info = "person (1-class)"
mean_count = "N/A"
total_train = "N/A"
total_val = "N/A"

if os.path.exists(config_path):
    with open(config_path) as f:
        cc = json.load(f)

    class_info = f"{', '.join(cc.get('class_names', ['person']))} ({cc.get('num_classes', 1)}-class)"
    total_train = cc.get('total_train_images', 'N/A')
    total_val = cc.get('total_val_images', 'N/A')

    if isinstance(cc.get('mean_count_per_image'), (int, float)):
        mean_count = f"{cc.get('mean_count_per_image'):.0f}"

# Override mean count using ground truth if available
gt_path = "/kaggle/working/ground_truth_counts.csv"
if os.path.exists(gt_path):
    df_gt = pd.read_csv(gt_path)
    if 'gt_count' in df_gt.columns:
        mean_count = f"{df_gt['gt_count'].mean():.0f}"

# Load tuning results safely
tuning_path = "/kaggle/working/tuning_results.json"

map50 = "N/A"
map50_95 = "N/A"
precision = "N/A"
recall = "N/A"
val_mae = "N/A"
best_conf = 0.35
best_iou = 0.45

if os.path.exists(tuning_path):
    with open(tuning_path) as f:
        t = json.load(f)

    if isinstance(t.get('map50'), (int, float)):
        map50 = f"{t.get('map50'):.4f}"

    if isinstance(t.get('map50_95'), (int, float)):
        map50_95 = f"{t.get('map50_95'):.4f}"

    if isinstance(t.get('precision'), (int, float)):
        precision = f"{t.get('precision'):.4f}"

    if isinstance(t.get('recall'), (int, float)):
        recall = f"{t.get('recall'):.4f}"

    if isinstance(t.get('best_mae'), (int, float)):
        val_mae = f"{t.get('best_mae'):.4f}"

    best_conf = t.get('best_conf', 0.35)
    best_iou = t.get('best_iou', 0.45)

# Detect trained models
model_s_exists = os.path.exists("/kaggle/working/runs/yolov8s_eduvision/weights/best.pt")
model_m_exists = os.path.exists("/kaggle/working/runs/yolov8m_eduvision/weights/best.pt")

if model_s_exists and model_m_exists:
    inference_mode = "YOLOv8s + YOLOv8m WBF Ensemble"
elif model_m_exists:
    inference_mode = "YOLOv8m Single Model"
else:
    inference_mode = "YOLOv8s Single Model"

# Auto date
today = datetime.now().strftime("%Y-%m-%d")

# ============================================================
# 2. BUILD REPORT
# ============================================================

report = f"""# EduVision 2026 — Technical Report
**Team:** GSM
**Date:** {today}
**Inference Mode:** {inference_mode}

---

## 1. Problem Approach

### Overall Strategy
We implemented a robust object detection pipeline using the YOLOv8 architecture. 
Given the dense classroom setting (avg. {mean_count} students per image), we optimized for small-object detection and occlusion handling.

### Architecture
* **Primary Model**: **YOLOv8s**
{"* **Secondary Model**: **YOLOv8m** for ensemble boosting." if model_m_exists else "* **Secondary Model**: Not trained (time-constrained)."}
{"* **Ensemble Method**: Weighted Boxes Fusion (WBF) + TTA." if model_m_exists else "* **Inference**: Optimized single-model threshold tuning."}
* **Classes**: `{class_info}`

---

## 2. Data Preprocessing & Augmentation

* **Annotation Format**: CSV → YOLO normalized format
* **Dataset Size**: {total_train} Train / {total_val} Val images
* **Coordinate Validation**: Clamped to valid normalized range
* **Tiny Box Filtering**: < 3px removed

### Augmentations
- Mosaic = 1.0
- Mixup = 0.15
- Copy-Paste = 0.1
- HSV Adjustments
- Scale = 0.5

---

## 3. Training Configuration

| Parameter | YOLOv8s | {"YOLOv8m" if model_m_exists else "N/A"} |
|-----------|----------|------------|
| Epochs | 50 | {"50" if model_m_exists else "—"} |
| Image Size | 640 | {"640" if model_m_exists else "—"} |
| Batch Size | 16 | {"8" if model_m_exists else "—"} |
| Optimizer | AdamW | {"SGD" if model_m_exists else "—"} |

---

## 4. Post Processing

- Confidence Threshold: {best_conf}
- IoU Threshold: {best_iou}
- Size Filtering Applied
{"- WBF Ensemble across models" if model_m_exists else "- Single Model NMS applied"}

---

## 5. Validation Results

| Metric | Score |
|--------|-------|
| mAP@0.5 | {map50} |
| mAP@0.5:0.95 | {map50_95} |
| Precision | {precision} |
| Recall | {recall} |
| Counting MAE | {val_mae} |

---

## 6. Deliverables

- count_predictions.csv
- detection_predictions.json
- best.pt weights
- Gradio Demo

---

## 7. Reproduction

```bash
pip install ultralytics ensemble-boxes pycocotools albumentations gradio
python train_model.py
python evaluate_and_tune.py
python inference_and_submission.py

"""

report_path = f"{OUTPUT_DIR}/technical_report.md"

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n✅ Technical report saved to: {report_path}")
print(f"📄 Size: {len(report)} characters")
print(f"📝 Word Count: {len(report.split())} words")
print("\n================ GENERATED REPORT PREVIEW ================\n")
print(report)