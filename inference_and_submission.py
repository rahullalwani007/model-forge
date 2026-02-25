
# EduVision 2026 - Inference + Submission

import os, json, glob, cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch

# AUTO-LOAD BEST THRESHOLDS FROM TUNING

tuning_path = "/kaggle/working/tuning_results.json"
if os.path.exists(tuning_path):
    with open(tuning_path) as f:
        tuning = json.load(f)
    CONF_THRESH = tuning['best_conf']
    IOU_THRESH  = tuning['best_iou']
    print(f"✅ Loaded tuned thresholds: conf={CONF_THRESH}, iou={IOU_THRESH}")
else:
    CONF_THRESH = 0.30
    IOU_THRESH  = 0.45
    print(f"⚠️  Using default thresholds: conf={CONF_THRESH}, iou={IOU_THRESH}")


# LOAD CLASS CONFIG

config_path = "/kaggle/working/class_config.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        class_config = json.load(f)
    CLASS_MODE = class_config['class_mode']
    STUDENT_CLASS_ID = class_config['student_class_id']
    TEACHER_CLASS_ID = class_config.get('teacher_class_id', None)
    NUM_CLASSES = class_config['num_classes']
    CLASS_NAMES = class_config['class_names']
else:
    CLASS_MODE = 'single'
    STUDENT_CLASS_ID = 0
    TEACHER_CLASS_ID = None
    NUM_CLASSES = 1
    CLASS_NAMES = ['person']

print(f"📋 Class mode: {CLASS_MODE} | Classes: {CLASS_NAMES}")
print(f"   Student class: {STUDENT_CLASS_ID} | Teacher class: {TEACHER_CLASS_ID}")


# CONFIG

MODEL1_PATH  = "/kaggle/working/runs/yolov8s_eduvision/weights/best.pt"
MODEL2_PATH  = "/kaggle/working/runs/yolov8m_eduvision/weights/best.pt"
TEST_IMG_DIR = "/kaggle/input/datasets/rahullalwani8/testing/test"
OUTPUT_DIR   = "/kaggle/working/submission"
IMG_SIZE     = 640
os.makedirs(OUTPUT_DIR, exist_ok=True)


# LOAD MODELS

USE_ENSEMBLE = os.path.exists(MODEL2_PATH)
print(f"\n📦 Model 1 (YOLOv8s): {'✅ Found' if os.path.exists(MODEL1_PATH) else '❌ Missing'}")
print(f"📦 Model 2 (YOLOv8m): {'✅ Found — ENSEMBLE MODE' if USE_ENSEMBLE else '⚠️  Not found — Single model'}")

model1 = YOLO(MODEL1_PATH)
model2 = None
wbf_available = False

if USE_ENSEMBLE:
    try:
        from ensemble_boxes import weighted_boxes_fusion
        model2 = YOLO(MODEL2_PATH)
        wbf_available = True
        print("✅ WBF Ensemble ready!")
    except ImportError:
        print("⚠️  ensemble-boxes not installed — pip install ensemble-boxes")
        USE_ENSEMBLE = False

DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# SINGLE MODEL INFERENCE

def predict_single(img_path, model, conf, iou, img_size=640):
    """Single-model prediction with class-aware output"""
    results = model(
        img_path, imgsz=img_size, conf=conf, iou=iou,
        verbose=False, device=DEVICE
    )
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            boxes.append([float(x1), float(y1), float(x2), float(y2), score, cls_id])
    return boxes



# WBF ENSEMBLE WITH TTA (class-aware)

def predict_ensemble_wbf(img_path, models, conf, iou, img_size=640):
    """WBF ensemble with TTA — preserves class information"""
    img = cv2.imread(img_path)
    if img is None:
        return []
    h, w = img.shape[:2]

    all_boxes, all_scores, all_labels = [], [], []

    for model in models:
        for flip in [False, True]:  
            img_input = img.copy()
            if flip:
                img_input = cv2.flip(img_input, 1)

            results = model(
                img_input, imgsz=img_size, conf=conf,
                iou=iou, verbose=False, device=DEVICE
            )

            boxes_norm, scores, labels = [], [], []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    sc = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    if flip:
                        x1, x2 = w - x2, w - x1
                    boxes_norm.append([
                        max(0, x1/w), max(0, y1/h),
                        min(1, x2/w), min(1, y2/h)
                    ])
                    scores.append(sc)
                    labels.append(cls_id)

            all_boxes.append(boxes_norm)
            all_scores.append(scores)
            all_labels.append(labels)

    if all(len(b) == 0 for b in all_boxes):
        return []

    boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        iou_thr=0.55, skip_box_thr=0.001
    )

    final = []
    for box, score, label in zip(boxes_wbf, scores_wbf, labels_wbf):
        final.append([
            box[0]*w, box[1]*h,
            box[2]*w, box[3]*h,
            float(score),
            int(round(label))  
        ])
    return final



# SIZE-BASED POST-PROCESSING FILTER

def filter_detections(boxes, img_h, img_w,
                       min_height_ratio=0.03,
                       min_width_ratio=0.01,
                       max_height_ratio=0.95):
    """Remove impossible detections"""
    filtered = []
    for box in boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        bh = y2 - y1
        bw = x2 - x1
        if (bh / img_h >= min_height_ratio and
            bw / img_w >= min_width_ratio and
            bh / img_h <= max_height_ratio and
            bw / img_w <= max_height_ratio):
            filtered.append(box)
    return filtered


# MAIN INFERENCE LOOP

test_images = sorted(
    glob.glob(f"{TEST_IMG_DIR}/*.jpg") +
    glob.glob(f"{TEST_IMG_DIR}/*.png") +
    glob.glob(f"{TEST_IMG_DIR}/*.jpeg")
)

print(f"\n🔍 Running inference on {len(test_images)} test images...")
mode = "YOLOv8s + YOLOv8m WBF Ensemble + TTA" if (USE_ENSEMBLE and wbf_available) else "YOLOv8s Single Model"
print(f"   Mode: {mode}")
print(f"   Conf: {CONF_THRESH} | IoU: {IOU_THRESH}")

submission_records = []
coco_predictions   = []

for idx, img_path in enumerate(test_images):
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    img_name = os.path.basename(img_path)
    img_stem = os.path.splitext(img_name)[0]

    # ---- Run detection ----
    if USE_ENSEMBLE and wbf_available:
        boxes = predict_ensemble_wbf(
            img_path, [model1, model2],
            conf=CONF_THRESH, iou=IOU_THRESH, img_size=IMG_SIZE
        )
    else:
        boxes = predict_single(
            img_path, model1,
            conf=CONF_THRESH, iou=IOU_THRESH, img_size=IMG_SIZE
        )

    # ---- Post-process ----
    boxes = filter_detections(boxes, h, w)

    # ---- Count ONLY students (class 0) for MAE metric ----
    student_count = 0
    teacher_count = 0

    for box in boxes:
        x1, y1, x2, y2, score = box[0], box[1], box[2], box[3], box[4]
        cls_id = box[5] if len(box) > 5 else 0

        coco_predictions.append({
            "image_id": img_stem,
            "category_id": int(cls_id) + 1,  
            "bbox": [round(x1, 2), round(y1, 2),
                     round(x2-x1, 2), round(y2-y1, 2)],
            "score": round(score, 4)
        })

        if cls_id == STUDENT_CLASS_ID:
            student_count += 1
        else:
            teacher_count += 1

    # ---- For MAE: count ONLY students ----
    if CLASS_MODE == 'multi':
        final_count = student_count
    else:
        final_count = student_count + teacher_count  # single class = all are students

    submission_records.append({
        "image_name": img_name,
        "predicted_count": final_count,
        "student_count": student_count,
        "teacher_count": teacher_count,
        "total_detections": len(boxes),
    })

    if (idx + 1) % 50 == 0 or idx == 0 or idx == len(test_images) - 1:
        extra = f" (students={student_count}, teachers={teacher_count})" if CLASS_MODE == 'multi' else ""
        print(f"  [{idx+1}/{len(test_images)}] {img_name} → {final_count} students{extra}")


# SAVE OUTPUTS

df_sub = pd.DataFrame(submission_records)

# Save the competition submission (just image_name + predicted_count)
df_submit = df_sub[['image_name', 'predicted_count']].copy()
df_submit.to_csv(f"{OUTPUT_DIR}/count_predictions.csv", index=False)
print(f"\n✅ Count predictions saved! ({len(df_submit)} images)")

# Save detailed version for analysis
df_sub.to_csv(f"{OUTPUT_DIR}/count_predictions_detailed.csv", index=False)

print(f"\n📊 COUNTING STATISTICS:")
print(df_sub['predicted_count'].describe())

if CLASS_MODE == 'multi':
    print(f"\n👥 Student counts: mean={df_sub['student_count'].mean():.2f}")
    print(f"🧑‍🏫 Teacher counts: mean={df_sub['teacher_count'].mean():.2f}")

# Save detection predictions
with open(f"{OUTPUT_DIR}/detection_predictions.json", "w") as f:
    json.dump(coco_predictions, f, indent=2)
print(f"\n✅ Detection predictions saved! ({len(coco_predictions)} total boxes)")

print("\n🎉 INFERENCE COMPLETE!")
print(f"   Submission files in: {OUTPUT_DIR}/")