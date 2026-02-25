
# EduVision 2026 - Evaluate + Tune Confidence Threshold


import os, json, glob, cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch

# ============================================================
# LOAD CONFIG
# ============================================================
config_path = "/kaggle/working/class_config.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        class_config = json.load(f)
    CLASS_MODE = class_config['class_mode']
    STUDENT_CLASS_ID = class_config['student_class_id']
    NUM_CLASSES = class_config['num_classes']
else:
    CLASS_MODE = 'single'
    STUDENT_CLASS_ID = 0
    NUM_CLASSES = 1

print(f"📋 Class mode: {CLASS_MODE} | Student class ID: {STUDENT_CLASS_ID}")

# ============================================================
# LOAD MODELS
# ============================================================
MODEL1_PATH = "/kaggle/working/runs/yolov8s_eduvision/weights/best.pt"
MODEL2_PATH = "/kaggle/working/runs/yolov8m_eduvision/weights/best.pt"
VAL_IMG_DIR = "/kaggle/working/yolo_dataset/images/val"
GT_CSV      = "/kaggle/working/ground_truth_counts.csv"

model1 = YOLO(MODEL1_PATH)
USE_MODEL2 = os.path.exists(MODEL2_PATH)
if USE_MODEL2:
    model2 = YOLO(MODEL2_PATH)
    print(f"✅ Model 1 + Model 2 loaded (ensemble tuning)")
else:
    print(f"✅ Model 1 loaded (single model tuning)")

# ============================================================
# LOAD GROUND TRUTH
# ============================================================
df_gt = pd.read_csv(GT_CSV)
df_val = df_gt[df_gt['split'] == 'val'].copy()
print(f"Validation set: {len(df_val)} images")

# ============================================================
# HELPER: Count students from YOLO results
# ============================================================
def count_students_from_results(results):
    """Count only student detections (class 0) from YOLO results"""
    student_count = 0
    total_count = 0
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            total_count += 1
            if cls_id == STUDENT_CLASS_ID:
                student_count += 1
    if CLASS_MODE == 'single':
        return total_count  # In single-class mode, all are "students"
    return student_count


def count_all_from_results(results):
    """Count all detections"""
    return sum(len(r.boxes) for r in results)


# ============================================================
# TUNE CONFIDENCE THRESHOLD
# ============================================================
conf_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
val_images = sorted(
    glob.glob(f"{VAL_IMG_DIR}/*.jpg") +
    glob.glob(f"{VAL_IMG_DIR}/*.png") +
    glob.glob(f"{VAL_IMG_DIR}/*.jpeg")
)

print(f"\n🔍 Tuning confidence threshold on {len(val_images)} val images...")

best_conf = 0.25
best_mae  = float('inf')
results_log = []

for conf in conf_thresholds:
    preds = []
    gts   = []

    for img_path in val_images:
        img_name = os.path.basename(img_path)

        # Get GT count (student count, not total)
        gt_row = df_val[
            df_val['file_name'].apply(lambda x: os.path.basename(x)) == img_name
        ]
        if len(gt_row) == 0:
            continue
        gt_count = int(gt_row.iloc[0]['gt_count'])  # This is student count

        # Run Model 1
        r1 = model1(img_path, imgsz=640, conf=conf, iou=0.45, verbose=False,
                     device=0 if torch.cuda.is_available() else 'cpu')
        c1 = count_students_from_results(r1)

        if USE_MODEL2:
            r2 = model2(img_path, imgsz=640, conf=conf, iou=0.45, verbose=False,
                        device=0 if torch.cuda.is_available() else 'cpu')
            c2 = count_students_from_results(r2)
            pred_count = round((c1 + c2) / 2)
        else:
            pred_count = c1

        preds.append(pred_count)
        gts.append(gt_count)

    if len(preds) == 0:
        continue

    mae = np.mean(np.abs(np.array(preds) - np.array(gts)))
    rmse = np.sqrt(np.mean((np.array(preds) - np.array(gts))**2))
    results_log.append({'conf': conf, 'mae': mae, 'rmse': rmse})
    print(f"  conf={conf:.2f} → MAE={mae:.4f} | RMSE={rmse:.4f}")

    if mae < best_mae:
        best_mae  = mae
        best_conf = conf

print(f"\n🎯 BEST confidence threshold: {best_conf} (MAE={best_mae:.4f})")

# ---- Also tune IoU threshold ----
print(f"\n🔍 Tuning IoU threshold (with best conf={best_conf})...")
iou_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
best_iou = 0.45
best_mae_iou = float('inf')

for iou in iou_thresholds:
    preds = []
    gts   = []

    for img_path in val_images:
        img_name = os.path.basename(img_path)
        gt_row = df_val[
            df_val['file_name'].apply(lambda x: os.path.basename(x)) == img_name
        ]
        if len(gt_row) == 0:
            continue
        gt_count = int(gt_row.iloc[0]['gt_count'])

        r1 = model1(img_path, imgsz=640, conf=best_conf, iou=iou, verbose=False,
                     device=0 if torch.cuda.is_available() else 'cpu')
        c1 = count_students_from_results(r1)

        if USE_MODEL2:
            r2 = model2(img_path, imgsz=640, conf=best_conf, iou=iou, verbose=False,
                        device=0 if torch.cuda.is_available() else 'cpu')
            c2 = count_students_from_results(r2)
            pred_count = round((c1 + c2) / 2)
        else:
            pred_count = c1

        preds.append(pred_count)
        gts.append(gt_count)

    if len(preds) == 0:
        continue

    mae = np.mean(np.abs(np.array(preds) - np.array(gts)))
    print(f"  iou={iou:.2f} → MAE={mae:.4f}")

    if mae < best_mae_iou:
        best_mae_iou = mae
        best_iou = iou

print(f"\n🎯 BEST IoU threshold: {best_iou} (MAE={best_mae_iou:.4f})")

# ============================================================
# RUN OFFICIAL YOLO VALIDATION (mAP)
# ============================================================
print("\n📊 Running official YOLO validation (mAP)...")
val_results = model1.val(
    data="/kaggle/working/eduvision.yaml",
    imgsz=640,
    conf=best_conf,
    iou=best_iou,
    device=0 if torch.cuda.is_available() else 'cpu',
    plots=True,
    save_json=True,
)

print(f"\n📈 VALIDATION METRICS:")
print(f"   mAP@0.5      : {val_results.box.map50:.4f}")
print(f"   mAP@0.5:0.95 : {val_results.box.map:.4f}")
print(f"   Precision     : {val_results.box.mp:.4f}")
print(f"   Recall        : {val_results.box.mr:.4f}")
print(f"   Counting MAE  : {best_mae_iou:.4f} (conf={best_conf}, iou={best_iou})")

# ---- Save best thresholds for inference ----
tuning_results = {
    'best_conf': float(best_conf),
    'best_iou': float(best_iou),
    'best_mae': float(best_mae_iou),
    'map50': float(val_results.box.map50),
    'map50_95': float(val_results.box.map),
    'precision': float(val_results.box.mp),
    'recall': float(val_results.box.mr),
    'class_mode': CLASS_MODE,
    'num_classes': NUM_CLASSES,
}
with open("/kaggle/working/tuning_results.json", "w") as f:
    json.dump(tuning_results, f, indent=2)

print(f"\n✅ Tuning results saved to /kaggle/working/tuning_results.json")
print(f"\n💡 USE THESE IN inference_and_submission.py:")
print(f"   CONF_THRESH = {best_conf}")
print(f"   IOU_THRESH  = {best_iou}")