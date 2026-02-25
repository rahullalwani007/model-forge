
# EduVision 2026 - Gradio Web App 


import subprocess
import sys


try:
    import gradio as gr
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio", "ultralytics"])
    import gradio as gr
    from ultralytics import YOLO

import cv2
import numpy as np
from PIL import Image
import os, json, tempfile, glob

print(f"📦 Gradio version: {gr.__version__}")


config_path = "/kaggle/working/class_config.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        class_config = json.load(f)
    CLASS_MODE = class_config['class_mode']
    STUDENT_CLASS_ID = class_config['student_class_id']
    NUM_CLASSES = class_config['num_classes']
    CLASS_NAMES = class_config['class_names']
else:
    CLASS_MODE = 'single'
    STUDENT_CLASS_ID = 0
    NUM_CLASSES = 1
    CLASS_NAMES = ['Person']


MODEL_S_PATH = "/kaggle/working/runs/yolov8s_eduvision/weights/best.pt"
MODEL_M_PATH = "/kaggle/working/runs/yolov8m_eduvision/weights/best.pt"

if os.path.exists(MODEL_S_PATH):
    MODEL_PATH = MODEL_S_PATH
elif os.path.exists(MODEL_M_PATH):
    MODEL_PATH = MODEL_M_PATH
else:
    print("⚠️  Custom model not found — using COCO pretrained YOLOv8s")
    MODEL_PATH = "yolov8s.pt"

model = YOLO(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")
print(f"📋 Class mode: {CLASS_MODE} | Classes: {CLASS_NAMES}")


tuning_path = "/kaggle/working/tuning_results.json"
DEFAULT_CONF = 0.30
DEFAULT_IOU  = 0.45
if os.path.exists(tuning_path):
    with open(tuning_path) as f:
        tuning = json.load(f)
    DEFAULT_CONF = tuning.get('best_conf', 0.30)
    DEFAULT_IOU  = tuning.get('best_iou',  0.45)
    print(f"✅ Loaded tuned thresholds: conf={DEFAULT_CONF}, iou={DEFAULT_IOU}")


COLORS = {
    0: (0, 255, 65),
    1: (255, 50, 50),
}

if CLASS_MODE == 'multi':
    LABELS = {0: 'Student', 1: 'Teacher'}
else:
    LABELS = {0: CLASS_NAMES[0] if CLASS_NAMES else 'Person'}



def detect_students(image, conf_threshold, iou_threshold):
    if image is None:
        return None, "❌ No image uploaded", ""

    img_rgb = np.array(image)

    # Handle grayscale or RGBA images
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        tmp_path = tmp.name

    results = model(
        tmp_path, imgsz=640,
        conf=float(conf_threshold),
        iou=float(iou_threshold),
        verbose=False
    )

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    h, w = img_bgr.shape[:2]
    annotated = img_rgb.copy()
    student_count = 0
    teacher_count = 0
    confidence_list = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            score = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            confidence_list.append(score)

            color = COLORS.get(cls_id, (255, 255, 0))
            label_name = LABELS.get(cls_id, f'Class {cls_id}')

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{label_name} {score:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-lh-6), (x1+lw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if cls_id == STUDENT_CLASS_ID:
                student_count += 1
            else:
                teacher_count += 1

    total = student_count + teacher_count

    if CLASS_MODE == 'multi':
        overlay_lines = [f"Students: {student_count}", f"Teachers: {teacher_count}"]
    else:
        overlay_lines = [f"People Detected: {total}"]

    overlay_h = 35 + (30 * len(overlay_lines))
    # Draw semi-transparent background for overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (320, overlay_h), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
    
    # Draw text on top
    for i, line in enumerate(overlay_lines):
        text_color = (0, 255, 65) if 'Student' in line or 'People' in line else (255, 80, 80)
        cv2.putText(annotated, line, (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    avg_conf = np.mean(confidence_list) if confidence_list else 0

    if CLASS_MODE == 'multi':
        stats = (
            f"📊 **Detection Report**\n\n"
            f"🎓 **Students: {student_count}**\n\n"
            f"🧑‍🏫 **Teachers: {teacher_count}**\n\n"
            f"👥 **Total People: {total}**\n\n"
            f"---\n\n"
            f"🎯 Avg Confidence: {avg_conf:.3f}\n\n"
            f"📐 Image Size: {w}×{h} px\n\n"
            f"⚙️ Conf Threshold: {conf_threshold}\n\n"
            f"🔗 IoU Threshold: {iou_threshold}\n\n"
            f"🤖 Model: {os.path.basename(MODEL_PATH)}\n\n"
            f"🏷️ Mode: Multi-class (Student + Teacher)"
        )
        count_badge = f"## 🎓 {student_count} {'Student' if student_count == 1 else 'Students'} | 🧑‍🏫 {teacher_count} {'Teacher' if teacher_count == 1 else 'Teachers'}"
    else:
        stats = (
            f"📊 **Detection Report**\n\n"
            f"👥 **People Detected: {total}**\n\n"
            f"🎯 Avg Confidence: {avg_conf:.3f}\n\n"
            f"📐 Image Size: {w}×{h} px\n\n"
            f"⚙️ Conf Threshold: {conf_threshold}\n\n"
            f"🔗 IoU Threshold: {iou_threshold}\n\n"
            f"🤖 Model: {os.path.basename(MODEL_PATH)}"
        )
        count_badge = f"## 👥 {total} {'Person' if total == 1 else 'People'} Detected"

    return annotated, count_badge, stats



example_imgs = []
example_search_paths = [
    "/kaggle/working/yolo_dataset/images/val",
    "/kaggle/working/yolo_dataset/images/test",
    "/kaggle/input/datasets/rahullalwani8/testing/test",
]
for search_path in example_search_paths:
    if os.path.exists(search_path):
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            found = glob.glob(f"{search_path}/{ext}")
            found = [f for f in found if not os.path.basename(f).startswith("._")]
            example_imgs.extend(found)
    if len(example_imgs) >= 6:
        break

print(f"📁 Found {len(example_imgs)} example images for demo")


# ============================================================
# MODERN CSS — Dark Glass-morphism + Neon Accents
# ============================================================
CUSTOM_CSS = """
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:     #0a0e1a;
    --bg-card:        rgba(16, 22, 40, 0.85);
    --bg-card-hover:  rgba(22, 30, 52, 0.95);
    --border-subtle:  rgba(99, 179, 237, 0.15);
    --border-glow:    rgba(99, 179, 237, 0.4);
    --neon-green:     #00ff88;
    --neon-blue:      #3b82f6;
    --neon-purple:    #a855f7;
    --neon-cyan:      #06b6d4;
    --text-primary:   #f0f4ff;
    --text-secondary: #94a3b8;
    --text-muted:     #4a5568;
    --gradient-hero:  linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 50%, #0a1628 100%);
    --shadow-card:    0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,179,237,0.08);
    --shadow-glow:    0 0 30px rgba(59,130,246,0.25);
    --radius-card:    16px;
    --radius-btn:     12px;
}

/* ── Global Reset ── */
* { box-sizing: border-box; }

body,
.gradio-container {
    background: var(--gradient-hero) !important;
    background-attachment: fixed !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1340px !important;
    margin: 0 auto !important;
    padding: 0 24px 48px !important;
}

/* ── Animated Background Orbs ── */
.gradio-container::before {
    content: '';
    position: fixed;
    top: -200px; left: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: orb1 12s ease-in-out infinite alternate;
}
.gradio-container::after {
    content: '';
    position: fixed;
    bottom: -200px; right: -200px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(168,85,247,0.07) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: orb2 15s ease-in-out infinite alternate;
}
@keyframes orb1 { from { transform: translate(0,0) scale(1); } to { transform: translate(80px,60px) scale(1.15); } }
@keyframes orb2 { from { transform: translate(0,0) scale(1); } to { transform: translate(-60px,-80px) scale(1.1); } }

/* ── Hero Header ── */
#hero-header {
    text-align: center;
    padding: 52px 24px 36px;
    position: relative;
    z-index: 1;
}
#hero-header .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3);
    color: var(--neon-cyan);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 999px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
    animation: fadeSlideDown 0.6s ease both;
}
#hero-header h1 {
    font-size: clamp(2rem, 5vw, 3.4rem) !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    margin: 0 0 12px !important;
    background: linear-gradient(135deg, #ffffff 30%, #93c5fd 70%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: fadeSlideDown 0.7s ease 0.1s both;
}
#hero-header .subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    font-weight: 400;
    animation: fadeSlideDown 0.7s ease 0.2s both;
}
#hero-header .meta-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-top: 20px;
    flex-wrap: wrap;
    animation: fadeSlideDown 0.7s ease 0.3s both;
}
#hero-header .meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    color: var(--text-secondary);
    font-size: 0.8rem;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 999px;
    backdrop-filter: blur(6px);
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), rgba(168,85,247,0.3), transparent);
    margin: 0 auto 36px;
    max-width: 800px;
    animation: fadeIn 1s ease 0.5s both;
}

/* ── Stats Bar ── */
#stats-bar {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 28px;
    z-index: 1;
    position: relative;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-card);
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    box-shadow: var(--shadow-card);
}
.stat-card:hover {
    transform: translateY(-3px);
    border-color: var(--border-glow);
    box-shadow: var(--shadow-card), var(--shadow-glow);
}
.stat-card .stat-icon { font-size: 1.6rem; margin-bottom: 4px; }
.stat-card .stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
}
.stat-card .stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

/* ── Main Panel Cards ── */
.panel-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-card);
    padding: 28px;
    backdrop-filter: blur(16px);
    box-shadow: var(--shadow-card);
    transition: border-color 0.25s ease;
    position: relative;
    z-index: 1;
}
.panel-card:hover { border-color: var(--border-glow); }

.panel-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--neon-cyan);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-subtle), transparent);
}

/* ── Image Upload Zone ── */
.image-upload-zone .upload-container,
.image-upload-zone [data-testid="image"] {
    border: 2px dashed rgba(59,130,246,0.3) !important;
    border-radius: 12px !important;
    background: rgba(59,130,246,0.04) !important;
    transition: border-color 0.25s, background 0.25s !important;
    min-height: 340px;
}
.image-upload-zone .upload-container:hover,
.image-upload-zone [data-testid="image"]:hover {
    border-color: rgba(59,130,246,0.6) !important;
    background: rgba(59,130,246,0.08) !important;
}

/* ── Output Image ── */
.output-image-zone [data-testid="image"] {
    border: 1px solid rgba(0,255,136,0.2) !important;
    border-radius: 12px !important;
    background: rgba(0,0,0,0.4) !important;
    min-height: 340px;
}

/* ── Slider Overhaul ── */
.slider-wrapper label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.slider-wrapper input[type=range] {
    accent-color: var(--neon-blue) !important;
    height: 6px;
}
.slider-wrapper .info {
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
}

/* ── Sliders Divider ── */
.sliders-row {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 16px;
    gap: 20px !important;
}

/* ── Buttons ── */
#detect-btn {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    border: none !important;
    border-radius: var(--radius-btn) !important;
    color: #fff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em;
    padding: 14px 32px !important;
    min-height: 52px !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.4), 0 0 0 1px rgba(124,58,237,0.2) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    cursor: pointer;
}
#detect-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(37,99,235,0.55), 0 0 0 1px rgba(124,58,237,0.35) !important;
}
#detect-btn:active { transform: translateY(0) !important; }

#clear-btn {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: var(--radius-btn) !important;
    color: var(--text-secondary) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    min-height: 52px !important;
    transition: background 0.15s, border-color 0.15s !important;
}
#clear-btn:hover {
    background: rgba(255,255,255,0.09) !important;
    border-color: rgba(255,255,255,0.25) !important;
    color: #fff !important;
}

/* ── Count Badge Display ── */
#count-display {
    background: linear-gradient(135deg, rgba(0,255,136,0.07) 0%, rgba(59,130,246,0.07) 100%);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-top: 12px;
}
#count-display h2, #count-display p {
    color: var(--text-primary) !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

/* ── Stats Report Box ── */
#stats-report {
    background: rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 12px;
    font-size: 0.88rem !important;
    line-height: 1.8 !important;
    color: var(--text-secondary) !important;
}
#stats-report strong { color: var(--text-primary) !important; }

/* ── Accordion ── */
.gr-accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-card) !important;
    backdrop-filter: blur(12px);
    margin-top: 20px !important;
    overflow: hidden;
}
.gr-accordion summary {
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    padding: 16px 20px !important;
    font-size: 0.9rem !important;
    background: transparent !important;
}
.gr-accordion summary:hover { color: var(--text-primary) !important; }

/* ── Tables inside Accordion ── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    color: var(--text-secondary);
}
thead tr th {
    background: rgba(59,130,246,0.08) !important;
    color: var(--neon-cyan) !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--border-subtle);
}
tbody tr { transition: background 0.15s; }
tbody tr:hover { background: rgba(255,255,255,0.03) !important; }
tbody tr td {
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: var(--text-secondary);
}
tbody tr td:first-child { color: var(--text-primary); font-weight: 500; }

/* ── Examples Section ── */
.examples-section-label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--neon-cyan) !important;
    margin: 32px 0 12px !important;
}
.gr-samples-table tr td { cursor: pointer; }
.gr-samples-table tr td img {
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
    transition: border-color 0.2s, transform 0.2s;
}
.gr-samples-table tr td img:hover {
    border-color: var(--border-glow);
    transform: scale(1.03);
}

/* ── Color Legend Chips ── */
.legend-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 4px;
}
.legend-green { background: rgba(0,255,136,0.12); border: 1px solid rgba(0,255,136,0.3); color: #00ff88; }
.legend-red   { background: rgba(255,50,50,0.12);  border: 1px solid rgba(255,50,50,0.3);  color: #ff6b6b; }

/* ── Section Separator ── */
.section-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.2), transparent);
    margin: 36px 0;
}

/* ── Keyframe Animations ── */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-14px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 12px rgba(0,255,136,0.25); }
    50%       { box-shadow: 0 0 28px rgba(0,255,136,0.55); }
}

/* ── Footer ── */
footer { display: none !important; }

/* ── Responsive ── */
@media (max-width: 768px) {
    #stats-bar { grid-template-columns: 1fr; }
    .gradio-container { padding: 0 12px 32px !important; }
    #hero-header h1 { font-size: 1.8rem !important; }
}
"""


try:
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="#0a0e1a",
        body_background_fill_dark="#0a0e1a",
        block_background_fill="rgba(16,22,40,0.85)",
        block_background_fill_dark="rgba(16,22,40,0.85)",
        block_border_color="rgba(99,179,237,0.15)",
        block_border_color_dark="rgba(99,179,237,0.15)",
        block_label_text_color="#94a3b8",
        block_label_text_color_dark="#94a3b8",
        block_title_text_color="#f0f4ff",
        block_title_text_color_dark="#f0f4ff",
        input_background_fill="rgba(10,14,26,0.7)",
        input_background_fill_dark="rgba(10,14,26,0.7)",
        input_border_color="rgba(99,179,237,0.2)",
        input_border_color_dark="rgba(99,179,237,0.2)",
        slider_color="#3b82f6",
        slider_color_dark="#3b82f6",
        button_primary_background_fill="linear-gradient(135deg,#2563eb,#7c3aed)",
        button_primary_background_fill_dark="linear-gradient(135deg,#2563eb,#7c3aed)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="rgba(255,255,255,0.05)",
        button_secondary_background_fill_dark="rgba(255,255,255,0.05)",
        button_secondary_text_color="#94a3b8",
        button_secondary_text_color_dark="#94a3b8",
        border_color_primary="rgba(99,179,237,0.2)",
        border_color_primary_dark="rgba(99,179,237,0.2)",
        shadow_drop="0 8px 32px rgba(0,0,0,0.5)",
        shadow_drop_lg="0 16px 48px rgba(0,0,0,0.6)",
        radius_lg="16px",
        radius_md="12px",
        radius_sm="8px",
    )
except Exception:
    try:
        theme = gr.themes.Default(primary_hue="blue")
    except Exception:
        theme = "default"



mode_label    = '🏷️ Multi-class (Student + Teacher)' if CLASS_MODE == 'multi' else '👥 Person Detection'
color_legend  = '| 🟢 Green = Student &nbsp; 🔴 Red = Teacher |' if CLASS_MODE == 'multi' else '| 🟢 Green = Person |'
model_name    = os.path.basename(MODEL_PATH)

with gr.Blocks(
    theme=theme,
    title="EduVision 2026 — Classroom Student Counter",
    css=CUSTOM_CSS,
) as demo:

    # ── Hero Header ──────────────────────────────────────────
    gr.HTML(f"""
    <div id="hero-header">
        <div class="hero-badge">
            <span>⚡</span> YOLOv8 · Real-Time Detection · 2026
        </div>
        <h1>EduVision 2026</h1>
        <p class="subtitle">AI-Powered Classroom Crowd Detection &amp; Counting</p>
        <div class="meta-row">
            <span class="meta-chip">🏷️ {mode_label}</span>
            <span class="meta-chip">🤖 {model_name}</span>
            <span class="meta-chip">⚙️ Conf {DEFAULT_CONF} · IoU {DEFAULT_IOU}</span>
            <span class="meta-chip">{color_legend}</span>
        </div>
    </div>
    <div class="hero-divider"></div>
    """)

    # ── Main Two-Column Layout ────────────────────────────────
    with gr.Row(equal_height=False, elem_classes=["main-row"]):

        # ── LEFT — Input Panel ────────────────────────────────
        with gr.Column(scale=1, elem_classes=["panel-card"]):
            gr.HTML('<div class="panel-title">📷 Input Image</div>')

            input_image = gr.Image(
                label="",
                type="pil",
                height=360,
                elem_classes=["image-upload-zone"],
                show_label=False,
            )

            # Sliders Row
            with gr.Row(elem_classes=["sliders-row"]):
                with gr.Column(elem_classes=["slider-wrapper"]):
                    conf_slider = gr.Slider(
                        minimum=0.05, maximum=0.9,
                        value=DEFAULT_CONF, step=0.05,
                        label="🎯 Confidence",
                        info="Higher = fewer but more certain detections",
                    )
                with gr.Column(elem_classes=["slider-wrapper"]):
                    iou_slider = gr.Slider(
                        minimum=0.1, maximum=0.9,
                        value=DEFAULT_IOU, step=0.05,
                        label="🔗 IoU (NMS)",
                        info="Lower = fewer overlapping boxes",
                    )

            # Action Buttons
            with gr.Row():
                detect_btn = gr.Button(
                    "🚀  Detect & Count",
                    variant="primary",
                    size="lg",
                    elem_id="detect-btn",
                )
                clear_btn = gr.ClearButton(
                    [input_image],
                    value="🗑️  Clear",
                    size="lg",
                    elem_id="clear-btn",
                )

        # ── RIGHT — Output Panel ──────────────────────────────
        with gr.Column(scale=1, elem_classes=["panel-card"]):
            gr.HTML('<div class="panel-title">🔍 Detection Results</div>')

            output_image = gr.Image(
                label="",
                height=360,
                elem_classes=["output-image-zone"],
                show_label=False,
            )

            count_output = gr.Markdown(
                value="<p style='text-align:center;color:#4a5568;font-size:0.95rem;padding:8px 0;'>Upload an image and click <strong style='color:#3b82f6'>Detect &amp; Count</strong></p>",
                elem_id="count-display",
            )

            stats_output = gr.Markdown(
                value="",
                elem_id="stats-report",
            )

    # ── Example Images ────────────────────────────────────────
    if example_imgs:
        gr.HTML('<div class="section-sep"></div>')
        gr.HTML('<div class="examples-section-label">📁 &nbsp; Try with Example Classroom Images</div>')
        gr.Examples(
            examples=[[img] for img in example_imgs[:6]],
            inputs=[input_image],
            label="",
        )

    # ── Info Accordions ───────────────────────────────────────
    gr.HTML('<div class="section-sep"></div>')

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("🤖  Model Information", open=False):
                class_info_str = (
                    "Student + Teacher (2-class)"
                    if CLASS_MODE == 'multi'
                    else f"{', '.join(CLASS_NAMES)} ({NUM_CLASSES}-class)"
                )
                gr.Markdown(f"""
| Property | Value |
|---|---|
| **Architecture** | YOLOv8s (Small) |
| **Pretrained On** | COCO |
| **Fine-tuned On** | EduVision 2026 Classroom Dataset |
| **Classes** | {class_info_str} |
| **Task** | Person Detection + Counting |
| **Tuned Conf** | `{DEFAULT_CONF}` |
| **Tuned IoU** | `{DEFAULT_IOU}` |
| **Model File** | `{model_name}` |
                """)

        with gr.Column(scale=1):
            with gr.Accordion("🎨  Color Legend", open=False):
                if CLASS_MODE == 'multi':
                    gr.HTML("""
                    <div style="padding:12px 0; display:flex; gap:12px; flex-wrap:wrap;">
                        <span class="legend-chip legend-green">🟢 &nbsp; Student</span>
                        <span class="legend-chip legend-red">🔴 &nbsp; Teacher / Instructor</span>
                    </div>
                    """)
                else:
                    gr.HTML("""
                    <div style="padding:12px 0;">
                        <span class="legend-chip legend-green">🟢 &nbsp; Person</span>
                    </div>
                    """)


    detect_btn.click(
        fn=detect_students,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, count_output, stats_output],
    )

    input_image.change(
        fn=detect_students,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, count_output, stats_output],
    )



if __name__ == "__main__":
    try:
        demo.launch(
            share=True,
            server_port=7860,
            show_error=True,
            quiet=False,
        )
    except Exception as e:
        print(f"⚠️  share=True failed ({e}), launching locally...")
        demo.launch(
            share=False,
            server_port=7860,
            show_error=True,
            quiet=False,
        )