# ============================================================
# EduVision 2026 - Step 1: Setup & EDA
# UPDATED: Supports CSV annotation format
# ============================================================

import subprocess
subprocess.run(["pip", "install", "-q", "ultralytics", "supervision", "pycocotools",
                "albumentations", "timm", "ensemble-boxes", "scipy", "gradio",
                "scikit-learn"])

import os, json, glob, shutil, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
from collections import Counter, defaultdict
import cv2

print("✅ All imports successful")

# ============================================================
# PATHS — UPDATE TO MATCH YOUR KAGGLE DATASET
# ============================================================
DATASET_ROOT = "/kaggle/input/human-detection-in-classroom"  # UPDATE THIS
OUTPUT_DIR   = "/kaggle/working/eduvision"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- AUTO-DETECT DATASET STRUCTURE ----
print("\n📁 DATASET STRUCTURE:")
for root, dirs, files in os.walk(DATASET_ROOT):
    level = root.replace(DATASET_ROOT, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 3:
        subindent = ' ' * 2 * (level + 1)
        for file in files[:8]:
            print(f'{subindent}{file}')
        if len(files) > 8:
            print(f'{subindent}... and {len(files)-8} more files')

# ---- COUNT FILES BY TYPE ----
json_files = glob.glob(f"{DATASET_ROOT}/**/*.json", recursive=True)
xml_files  = glob.glob(f"{DATASET_ROOT}/**/*.xml", recursive=True)
txt_files  = glob.glob(f"{DATASET_ROOT}/**/*.txt", recursive=True)
csv_files  = glob.glob(f"{DATASET_ROOT}/**/*.csv", recursive=True)
img_files  = (glob.glob(f"{DATASET_ROOT}/**/*.jpg", recursive=True) +
              glob.glob(f"{DATASET_ROOT}/**/*.png", recursive=True) +
              glob.glob(f"{DATASET_ROOT}/**/*.jpeg", recursive=True))

print(f"\n📊 DATASET FILE COUNTS:")
print(f"  Images : {len(img_files)}")
print(f"  CSV    : {len(csv_files)}")
print(f"  JSON   : {len(json_files)}")
print(f"  XML    : {len(xml_files)}")
print(f"  TXT    : {len(txt_files)}")

# ---- FIND AND LOAD CSV ANNOTATIONS ----
ann_csv = None
for cf in csv_files:
    try:
        df_test = pd.read_csv(cf, nrows=5)
        # Check if it looks like annotation CSV
        cols_lower = [c.lower() for c in df_test.columns]
        if any('xmin' in c or 'x_min' in c or 'bbox' in c for c in cols_lower):
            ann_csv = cf
            print(f"\n✅ Found annotation CSV: {cf}")
            break
        elif 'class' in cols_lower and 'filename' in cols_lower:
            ann_csv = cf
            print(f"\n✅ Found annotation CSV: {cf}")
            break
    except:
        pass

# Also check for COCO JSON
ann_json = None
for jf in json_files:
    try:
        with open(jf) as f:
            data = json.load(f)
        if 'annotations' in data and 'images' in data:
            ann_json = jf
            print(f"\n✅ Found COCO JSON: {jf}")
            break
    except:
        pass

# ---- ANALYZE BASED ON FORMAT FOUND ----
if ann_csv:
    print(f"\n📋 ANNOTATION FORMAT: CSV")
    df = pd.read_csv(ann_csv)
    print(f"\n📊 CSV INFO:")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Total rows : {len(df)}")
    print(f"\n  First 5 rows:")
    print(df.head().to_string())

    # Standardize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ['filename', 'file_name', 'image']:
            col_map[col] = 'filename'
        elif cl in ['class', 'label', 'class_name']:
            col_map[col] = 'class'
        elif cl in ['xmin', 'x_min']:
            col_map[col] = 'xmin'
        elif cl in ['ymin', 'y_min']:
            col_map[col] = 'ymin'
        elif cl in ['xmax', 'x_max']:
            col_map[col] = 'xmax'
        elif cl in ['ymax', 'y_max']:
            col_map[col] = 'ymax'
        elif cl in ['width']:
            col_map[col] = 'width'
        elif cl in ['height']:
            col_map[col] = 'height'
    df = df.rename(columns=col_map)

    # Class analysis
    print(f"\n🏷️  CLASSES:")
    classes = df['class'].unique()
    for cls in classes:
        count = len(df[df['class'] == cls])
        print(f"   '{cls}': {count} annotations")

    has_student = any('student' in str(c).lower() for c in classes)
    has_teacher = any('teacher' in str(c).lower() or 'instructor' in str(c).lower() for c in classes)

    if has_student and has_teacher:
        print("\n  🎯 MULTI-CLASS: Student + Teacher found!")
    elif any('person' in str(c).lower() for c in classes):
        print("\n  📌 SINGLE-CLASS: Only 'Person' found")
        print("  → All detections count as students for MAE")
    else:
        print(f"\n  ⚠️  Classes: {classes}")

    # Count per image
    unique_imgs = df['filename'].unique()
    counts_per_image = df.groupby('filename').size()
    counts = counts_per_image.values

    print(f"\n👥 PEOPLE PER IMAGE:")
    print(f"  Unique images: {len(unique_imgs)}")
    print(f"  Min   : {counts.min()}")
    print(f"  Max   : {counts.max()}")
    print(f"  Mean  : {counts.mean():.2f}")
    print(f"  Median: {np.median(counts):.1f}")
    print(f"  Std   : {np.std(counts):.2f}")

    # Bounding box analysis
    if all(c in df.columns for c in ['xmin', 'ymin', 'xmax', 'ymax']):
        df['bbox_w'] = df['xmax'] - df['xmin']
        df['bbox_h'] = df['ymax'] - df['ymin']
        df['bbox_area'] = df['bbox_w'] * df['bbox_h']

        print(f"\n📐 BOUNDING BOX ANALYSIS:")
        print(f"  Avg width  : {df['bbox_w'].mean():.0f} px")
        print(f"  Avg height : {df['bbox_h'].mean():.0f} px")
        print(f"  Avg area   : {df['bbox_area'].mean():.0f} px²")
        print(f"  Min area   : {df['bbox_area'].min():.0f} px²")
        print(f"  Max area   : {df['bbox_area'].max():.0f} px²")

    # ---- VISUALIZE ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Count distribution
    axes[0, 0].hist(counts, bins=30, color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of People per Image', fontsize=14)
    axes[0, 0].set_xlabel('Number of People')
    axes[0, 0].set_ylabel('Number of Images')
    axes[0, 0].axvline(counts.mean(), color='red', linestyle='--',
                        label=f'Mean={counts.mean():.1f}')
    axes[0, 0].legend()

    # BBox area distribution
    if 'bbox_area' in df.columns:
        axes[0, 1].hist(df['bbox_area'], bins=50, color='coral', edgecolor='black')
        axes[0, 1].set_title('Bounding Box Area Distribution', fontsize=14)
        axes[0, 1].set_xlabel('Area (px²)')

    # Class distribution
    class_counts = df['class'].value_counts()
    axes[1, 0].bar(class_counts.index, class_counts.values, color='mediumpurple')
    axes[1, 0].set_title('Annotations per Class', fontsize=14)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Sample image with boxes
    sample_img_name = random.choice(unique_imgs[:50])
    sample_anns = df[df['filename'] == sample_img_name]
    img_dir = os.path.dirname(ann_csv)
    img_path = os.path.join(img_dir, sample_img_name)

    if not os.path.exists(img_path):
        # Search for it
        matches = glob.glob(f"{DATASET_ROOT}/**/{sample_img_name}", recursive=True)
        if matches:
            img_path = matches[0]

    if os.path.exists(img_path):
        img = plt.imread(img_path)
        axes[1, 1].imshow(img)
        for _, ann in sample_anns.iterrows():
            x1, y1 = ann['xmin'], ann['ymin']
            w, h = ann['xmax'] - ann['xmin'], ann['ymax'] - ann['ymin']
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                      edgecolor='lime', facecolor='none')
            axes[1, 1].add_patch(rect)
        axes[1, 1].set_title(f'Sample: {len(sample_anns)} people', fontsize=14)
        axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'Image not found', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_analysis.png", dpi=150)
    plt.show()
    print(f"\n✅ EDA chart saved to {OUTPUT_DIR}/eda_analysis.png")

    # Save EDA info
    eda_info = {
        'annotation_format': 'csv',
        'csv_path': ann_csv,
        'num_images': int(len(unique_imgs)),
        'num_annotations': int(len(df)),
        'classes': list(classes),
        'has_multiclass': has_student and has_teacher,
        'mean_count': float(counts.mean()),
        'max_count': int(counts.max()),
    }
    with open(f"{OUTPUT_DIR}/eda_info.json", "w") as f:
        json.dump(eda_info, f, indent=2)
    print(f"✅ EDA info saved")

elif ann_json:
    print("📋 ANNOTATION FORMAT: COCO JSON — use original convert_to_yolo.py")

else:
    print("\n⚠️ No annotation file detected!")
    print("   Check dataset structure manually.")