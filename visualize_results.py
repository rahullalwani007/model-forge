
# EduVision 2026 - Visualize Results

import os, glob, cv2, random, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import pandas as pd

# Load config
config_path = "/kaggle/working/class_config.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        class_config = json.load(f)
    CLASS_MODE = class_config['class_mode']
    CLASS_NAMES = class_config['class_names']
    STUDENT_CLASS_ID = class_config['student_class_id']
else:
    CLASS_MODE = 'single'
    CLASS_NAMES = ['person']
    STUDENT_CLASS_ID = 0

# ============================================================
# MODEL SELECTION — Use best available model
# ============================================================
MODEL_S_PATH = "/kaggle/working/runs/yolov8s_eduvision/weights/best.pt"
MODEL_M_PATH = "/kaggle/working/runs/yolov8m_eduvision/weights/best.pt"

# Use YOLOv8s as primary (it was tuned on). Use m only if s doesn't exist.
if os.path.exists(MODEL_S_PATH):
    MODEL_PATH = MODEL_S_PATH
    print(f"✅ Using YOLOv8s model (primary, tuned)")
elif os.path.exists(MODEL_M_PATH):
    MODEL_PATH = MODEL_M_PATH
    print(f"✅ Using YOLOv8m model (fallback)")
else:
    raise FileNotFoundError("❌ No trained model found!")

TEST_IMG_DIR = "/kaggle/input/datasets/rahullalwani8/testing/test"
OUTPUT_DIR   = "/kaggle/working/submission/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tuned thresholds (auto from evaluate_and_tune.py)
tuning_path = "/kaggle/working/tuning_results.json"
CONF = 0.30
IOU = 0.45
if os.path.exists(tuning_path):
    with open(tuning_path) as f:
        tuning = json.load(f)
    CONF = tuning.get('best_conf', 0.30)
    IOU = tuning.get('best_iou', 0.45)
    print(f"✅ Loaded tuned thresholds: conf={CONF}, iou={IOU}")
else:
    print(f"⚠️  No tuning file found — using defaults: conf={CONF}, iou={IOU}")

model = YOLO(MODEL_PATH)

COLORS = {0: '#00FF41', 1: '#FF3232'}


raw_files = (glob.glob(f"{TEST_IMG_DIR}/*.jpg") +
             glob.glob(f"{TEST_IMG_DIR}/*.png") +
             glob.glob(f"{TEST_IMG_DIR}/*.jpeg"))

test_images = [f for f in raw_files
               if not os.path.basename(f).startswith("._")
               and os.path.getsize(f) > 1000]  # Also skip corrupt tiny files

print(f"📂 Found {len(raw_files)} files, {len(test_images)} valid images (filtered {len(raw_files)-len(test_images)} ghost/corrupt files)")

if len(test_images) == 0:
    print("❌ No test images found! Check TEST_IMG_DIR path.")
    print(f"   Looking in: {TEST_IMG_DIR}")
    print(f"   Exists: {os.path.exists(TEST_IMG_DIR)}")
    if os.path.exists(TEST_IMG_DIR):
        print(f"   Contents: {os.listdir(TEST_IMG_DIR)[:10]}")
else:
    n_vis = min(12, len(test_images))
    print(f"Visualizing {n_vis} sample predictions...")

    rows = max(1, min(3, (n_vis + 3) // 4))
    cols = max(1, min(4, n_vis))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))

    # Handle single subplot edge case
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    sample_imgs = random.sample(test_images, n_vis)

    for idx, img_path in enumerate(sample_imgs):
        img_bgr = cv2.imread(img_path)

        # FIX: Skip if cv2 can't read the image (extra safety)
        if img_bgr is None:
            print(f"  ⚠️  Could not read: {os.path.basename(img_path)}, skipping")
            axes[idx].text(0.5, 0.5, 'Could not load', ha='center', va='center')
            axes[idx].axis('off')
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        results = model(img_path, imgsz=640, conf=CONF, iou=IOU, verbose=False)

        ax = axes[idx]
        ax.imshow(img_rgb)

        student_count = 0
        teacher_count = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                color = COLORS.get(cls_id, '#FFFF00')

                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'Cls{cls_id}'
                ax.text(x1, y1-3, f'{label} {score:.2f}', fontsize=6,
                        color=color, fontweight='bold')

                if cls_id == STUDENT_CLASS_ID:
                    student_count += 1
                else:
                    teacher_count += 1

        if CLASS_MODE == 'multi':
            title = f'{os.path.basename(img_path)}\n🎓{student_count} 🧑‍🏫{teacher_count}'
        else:
            title = f'{os.path.basename(img_path)}\nCount: {student_count + teacher_count}'

        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')

    # Hide unused axes
    for j in range(n_vis, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('EduVision 2026 — Detection Visualization',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sample_predictions.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Visualization grid saved!")

    # ---- Save individual annotated images ----
    saved_count = 0
    for img_path in test_images[:20]:
        img_check = cv2.imread(img_path)
        if img_check is None:
            continue
        results = model(img_path, imgsz=640, conf=CONF, iou=IOU, verbose=False)
        for r in results:
            annotated = r.plot()
            out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
            cv2.imwrite(out_path, annotated)
            saved_count += 1

    print(f"✅ {saved_count} individual annotated images saved to {OUTPUT_DIR}")