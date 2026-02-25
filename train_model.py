
# EduVision 2026 - Step 3: Train YOLOv8s + YOLOv8m
# YOLOv8s: ~30-45 min | YOLOv8m: ~50-70 min on T4

import os, json, time, glob
import torch
from ultralytics import YOLO

print(f"🚀 GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

YAML_PATH  = "/kaggle/working/eduvision.yaml"
OUTPUT_DIR = "/kaggle/working/runs"

# ---- Load class config ----
config_path = "/kaggle/working/class_config.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        class_config = json.load(f)
    NUM_CLASSES = class_config['num_classes']
    CLASS_NAMES = class_config['class_names']
    print(f"📋 Class config: {NUM_CLASSES} classes → {CLASS_NAMES}")
else:
    print("⚠️  No class_config.json found — using defaults from YAML")


# MODEL 1: YOLOv8s — PRIMARY (ALWAYS TRAIN THIS)

print("\n" + "="*60)
print("🏋️  TRAINING MODEL 1: YOLOv8s (Primary — Fast & Accurate)")
print("="*60)

start_time = time.time()

model1 = YOLO('yolov8s.pt')

results1 = model1.train(
    data=YAML_PATH,
    epochs=50,
    imgsz=640,
    batch=16,
    workers=4,
    device=0 if torch.cuda.is_available() else 'cpu',
    project=OUTPUT_DIR,
    name="yolov8s_eduvision",

    # Optimizer
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,

    # Augmentation — tuned for classroom diversity
    augment=True,
    mosaic=1.0,
    mixup=0.15,          
    copy_paste=0.1,
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0001,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    # Detection
    conf=0.25,
    iou=0.45,

    # Checkpointing
    save=True,
    save_period=10,
    patience=15,

    # Logging
    plots=True,
    verbose=True,
)

model1_time = (time.time() - start_time) / 60
print(f"\n✅ Model 1 Training Complete! ({model1_time:.1f} min)")
print(f"   Best weights: {OUTPUT_DIR}/yolov8s_eduvision/weights/best.pt")


# MODEL 2: YOLOv8m — ENSEMBLE 

estimated_model2_time = model1_time * 2.0
HACKATHON_TOTAL_HOURS = 5
elapsed_hours = model1_time / 60
remaining_hours = HACKATHON_TOTAL_HOURS - elapsed_hours

SAFETY_BUFFER_HOURS = 1.5
can_train_model2 = (remaining_hours - estimated_model2_time / 60) > SAFETY_BUFFER_HOURS

print(f"\n⏱️  TIME CHECK:")
print(f"   Model 1 took: {model1_time:.1f} min")
print(f"   Estimated Model 2: {estimated_model2_time:.0f} min")
print(f"   Remaining time: {remaining_hours * 60:.0f} min")
print(f"   Safety buffer needed: {SAFETY_BUFFER_HOURS * 60:.0f} min")

if can_train_model2:
    print("\n" + "="*60)
    print("🏋️  TRAINING MODEL 2: YOLOv8m (Ensemble — Bonus Accuracy)")
    print("="*60)

    start_time2 = time.time()
    model2 = YOLO('yolov8m.pt')

    results2 = model2.train(
        data=YAML_PATH,
        epochs=20,
        imgsz=640,
        batch=8,
        workers=4,
        device=0 if torch.cuda.is_available() else 'cpu',
        project=OUTPUT_DIR,
        name="yolov8m_eduvision",

        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,

        augment=True,
        mosaic=0.9,
        mixup=0.05,
        copy_paste=0.05,
        degrees=5,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        conf=0.25,
        iou=0.45,

        save=True,
        patience=15,
        plots=True,
        verbose=True,
    )

    model2_time = (time.time() - start_time2) / 60
    print(f"\n✅ Model 2 Training Complete! ({model2_time:.1f} min)")
    print(f"\n🎯 BOTH MODELS TRAINED — ENSEMBLE MODE ACTIVATED!")

else:
    print(f"\n⚠️  Not enough time for Model 2 — SKIPPING.")
    print(f"   Proceeding with YOLOv8s only. Still very competitive! 💪")

print(f"\n📦 Available model weights:")
for wpath in glob.glob(f"{OUTPUT_DIR}/*/weights/best.pt"):
    size_mb = os.path.getsize(wpath) / 1e6
    print(f"   {wpath} ({size_mb:.1f} MB)")

