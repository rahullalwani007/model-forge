
# EduVision 2026 - Step 2: Auto-Convert CSV to YOLO

import os, shutil, glob, json, cv2
import pandas as pd
from sklearn.model_selection import train_test_split

print("🔍 Initializing Data Conversion...")
csv_files = glob.glob("/kaggle/input/**/*.csv", recursive=True)
CSV_ANN_PATH = csv_files[0]
YOLO_ROOT = "/kaggle/working/yolo_dataset"

for split in ['train', 'val', 'test']:
    os.makedirs(f"{YOLO_ROOT}/images/{split}", exist_ok=True)
    os.makedirs(f"{YOLO_ROOT}/labels/{split}", exist_ok=True)

df = pd.read_csv(CSV_ANN_PATH)
df['class_lower'] = df['class'].astype(str).str.lower().str.strip()


unique_classes = df['class_lower'].unique()
CLASS_MODE = 'single'
NUM_CLASSES = 1
CLASS_NAMES = ['person']
cat_mapping = {c: 0 for c in unique_classes}

print(f"📋 Mode: {CLASS_MODE} | Classes: {CLASS_NAMES}")

# Split Train/Val
unique_images = df['filename'].unique()
train_imgs, val_imgs = train_test_split(unique_images, test_size=0.1, random_state=42)
split_map = {img: 'train' for img in train_imgs}
split_map.update({img: 'val' for img in val_imgs})

# Find all images dynamically
print("🔍 Indexing image files... (this takes a few seconds)")
all_img_paths = glob.glob("/kaggle/input/**/*.jpg", recursive=True) + glob.glob("/kaggle/input/**/*.png", recursive=True)
img_path_dict = {os.path.basename(p): p for p in all_img_paths}

records = []
skipped = 0
grouped = df.groupby('filename')

print("\n🔄 Converting annotations and copying images...")
for fname, group in grouped:
    if fname not in img_path_dict:
        skipped += 1
        continue
        
    src_path = img_path_dict[fname]
    split = split_map.get(fname, 'train')
    
    dst_img = os.path.join(YOLO_ROOT, "images", split, fname)
    dst_lbl = os.path.join(YOLO_ROOT, "labels", split, os.path.splitext(fname)[0] + ".txt")
    
    if not os.path.exists(dst_img):
        shutil.copy2(src_path, dst_img)
        
    yolo_lines = []
    total_count = 0
    
    for _, row in group.iterrows():
        w_img, h_img = float(row['width']), float(row['height'])
        
        # Safety fallback if CSV width/height is empty
        if w_img == 0 or h_img == 0 or pd.isna(w_img):
            img_cv = cv2.imread(src_path)
            if img_cv is not None: h_img, w_img = img_cv.shape[:2]
            else: continue
            
        xmin, ymin = float(row['xmin']), float(row['ymin'])
        xmax, ymax = float(row['xmax']), float(row['ymax'])
        
        if (xmax - xmin) < 3 or (ymax - ymin) < 3: continue
            
        cx, cy = ((xmin + xmax) / 2.0) / w_img, ((ymin + ymax) / 2.0) / h_img
        w_box, h_box = (xmax - xmin) / w_img, (ymax - ymin) / h_img
        
        # Clamp bounds
        cx, cy = max(0.001, min(0.999, cx)), max(0.001, min(0.999, cy))
        w_box, h_box = max(0.001, min(0.999, w_box)), max(0.001, min(0.999, h_box))
        
        yolo_class = cat_mapping.get(row['class_lower'], 0)
        yolo_lines.append(f"{yolo_class} {cx:.6f} {cy:.6f} {w_box:.6f} {h_box:.6f}")
        total_count += 1
        
    with open(dst_lbl, 'w') as f:
        f.write("\n".join(yolo_lines))
        
    records.append({
        'image_id': fname, 'file_name': fname, 'split': split,
        'student_count': total_count, 'teacher_count': 0, 'total_count': total_count,
        'gt_count': total_count,
    })

print(f"✅ Converted {len(records)} images. Skipped {skipped} missing images.")

# Save Data Configs
pd.DataFrame(records).to_csv("/kaggle/working/ground_truth_counts.csv", index=False)

yaml_content = f"""# EduVision 2026 - YOLOv8 Config
path: {YOLO_ROOT}
train: images/train
val: images/val
test: images/test

nc: {NUM_CLASSES}
names:
  0: person
"""
with open("/kaggle/working/eduvision.yaml", "w") as f:
    f.write(yaml_content)

with open("/kaggle/working/class_config.json", "w") as f:
    json.dump({'class_mode': CLASS_MODE, 'num_classes': NUM_CLASSES, 'class_names': CLASS_NAMES, 'student_class_id': 0}, f)

print(f"✅ YAML and Config written. DATA IS READY FOR TRAINING!")