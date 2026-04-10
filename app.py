"""
app.py — Streamlit Web Demo for Helmet & Seatbelt Detection

Run with:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Traffic Safety AI 🚦",
    page_icon="🚦",
    layout="wide",
)
st.title("Smart Traffic Safety AI 🚦")
st.caption("AI-powered Helmet & Safety Detection System | By Shravani Patil")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .metric-box {
        background: #1a1d27;
        border: 1px solid #2e3347;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-label { font-size: 13px; color: #8b92a5; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #ffffff; }
    .violation-badge {
        background: #2d1515;
        border: 1px solid #c0392b;
        border-radius: 6px;
        color: #e74c3c;
        padding: 4px 12px;
        font-size: 13px;
        font-weight: 600;
    }
    .safe-badge {
        background: #122215;
        border: 1px solid #27ae60;
        border-radius: 6px;
        color: #2ecc71;
        padding: 4px 12px;
        font-size: 13px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🪖 Road Safety Violation Detector")
st.markdown("Detects **helmet** and **seatbelt** violations in images and videos using YOLOv8.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)
    model_choice = st.selectbox("Model size", ["YOLOv8n (fast)", "YOLOv8s (balanced)", "Custom (best.pt)"])
    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Upload an image or video\n2. Click Detect\n3. View annotated results\n4. Download output")
    st.divider()
    st.markdown("**Classes detected:**")
    st.markdown("🟢 Helmet worn\n🔴 No helmet\n🟢 Seatbelt on\n🔴 No seatbelt")


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(choice):
    model_map = {
        "YOLOv8n (fast)": "yolov8n.pt",
        "YOLOv8s (balanced)": "yolov8s.pt",
        "Custom (best.pt)": "models/best.pt",
    }
    path = model_map[choice]
    if not os.path.exists(path) and "Custom" in choice:
        st.warning("Custom model not found. Using YOLOv8n instead.")
        path = "yolov8n.pt"
    return YOLO(path)


model = load_model(model_choice)

# Class config
CLASS_NAMES = {0: "helmet", 1: "no_helmet", 2: "seatbelt", 3: "no_seatbelt"}
CLASS_COLORS = {
    "helmet":      (0, 200, 80),
    "no_helmet":   (0, 50, 220),
    "seatbelt":    (0, 200, 80),
    "no_seatbelt": (0, 50, 220),
}


def annotate_frame(frame, conf):
    results = model(frame, conf=conf, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = CLASS_NAMES.get(cls_id, model.names.get(cls_id, str(cls_id)))
        color = CLASS_COLORS.get(label, (200, 200, 200))
        is_viol = "no_" in label

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label.replace('_',' ').upper()} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y1 - 4, th + 6)
        cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
        cv2.putText(frame, text, (x1 + 4, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({"label": label, "conf": score, "violation": is_viol})

    return frame, detections


# ── File Upload ───────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷 Image", "🎬 Video"])

with tab1:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_img:
        col1, col2 = st.columns(2)
        img_array = np.frombuffer(uploaded_img.read(), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

        if st.button("🔍 Detect violations", type="primary"):
            with st.spinner("Analyzing..."):
                t0 = time.time()
                annotated, detections = annotate_frame(frame.copy(), conf_threshold)
                elapsed = time.time() - t0

            with col2:
                st.subheader("Detected")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

            violations = [d for d in detections if d["violation"]]
            safe = [d for d in detections if not d["violation"]]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total detections", len(detections))
            m2.metric("Violations", len(violations))
            m3.metric("Safe", len(safe))
            m4.metric("Inference time", f"{elapsed*1000:.0f} ms")

            if violations:
                st.error(f"⚠️ {len(violations)} violation(s) detected!")
                for v in violations:
                    st.write(f"  • {v['label'].replace('_',' ').title()} — {v['conf']:.0%} confidence")
            else:
                st.success("✅ No violations detected.")

            # Download button
            _, buf = cv2.imencode(".jpg", annotated)
            st.download_button("⬇️ Download result", buf.tobytes(),
                               file_name="detected.jpg", mime="image/jpeg")


with tab2:
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile.flush()

        st.info("Video uploaded. Click Detect to process all frames.")

        if st.button("🔍 Process video", type="primary"):
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path = tempfile.mktemp(suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            progress = st.progress(0, text="Processing frames...")
            preview = st.empty()
            stats_cols = st.columns(3)

            total_violations = 0
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                annotated, dets = annotate_frame(frame.copy(), conf_threshold)
                viols = sum(1 for d in dets if d["violation"])
                total_violations += viols
                writer.write(annotated)

                if frame_idx % 10 == 0 or frame_idx == total_frames:
                    progress.progress(frame_idx / total_frames,
                                      text=f"Frame {frame_idx}/{total_frames}")
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                  caption=f"Frame {frame_idx}", use_column_width=True)

            cap.release()
            writer.release()
            progress.empty()

            st.success(f"✅ Done! {frame_idx} frames processed. {total_violations} violation events.")

            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download annotated video", f.read(),
                                   file_name="detected_output.mp4", mime="video/mp4")
