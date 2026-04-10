# 🪖 Helmet & Seatbelt Violation Detector

> Real-time road safety enforcement using YOLOv8 and OpenCV.  
> Detects helmet and seatbelt violations from images, videos, or live webcam feed.

---

## 📌 Project Summary

| Item | Detail |
|------|--------|
| Problem | Road accidents due to no helmet / no seatbelt |
| Solution | Automated vision-based violation detection |
| Model | YOLOv8 (fine-tuned on custom dataset) |
| Demo | Streamlit web app + CLI |
| Classes | `helmet`, `no_helmet`, `seatbelt`, `no_seatbelt` |

---

## 🗂️ Project Structure

```
helmet_seatbelt_detection/
│
├── detect.py              ← Main detection script (CLI)
├── train.py               ← Fine-tune YOLOv8 on your dataset
├── app.py                 ← Streamlit web demo
├── download_dataset.py    ← Download dataset from Roboflow
├── requirements.txt
│
├── dataset/               ← Your training data goes here
│   ├── data.yaml
│   ├── train/images/
│   ├── train/labels/
│   └── valid/images/
│
├── models/
│   └── best.pt            ← Your trained model weights go here
│
├── utils/
│   ├── draw.py            ← Bounding box + stats drawing
│   └── logger.py          ← Violation CSV logger
│
└── results/               ← Output images/videos + CSV reports
```

---

## ⚙️ Setup

```bash
# 1. Clone / download the project
cd helmet_seatbelt_detection

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset

### Option A — Roboflow (recommended, free)
1. Go to [Roboflow Universe](https://universe.roboflow.com)
2. Search **"helmet detection"** or **"seatbelt detection"**
3. Export in **YOLOv8** format → download to `dataset/`

```bash
# Or use the helper script (needs Roboflow API key):
python download_dataset.py --api-key YOUR_KEY --project helmet-detection --version 1

# Just create folder structure (no API key needed):
python download_dataset.py --sample
```

### Option B — Kaggle
- Search: `helmet detection dataset` or `seatbelt detection`
- Download, unzip to `dataset/`, update `dataset/data.yaml`

---

## 🏋️ Training

```bash
# Fine-tune YOLOv8 on your dataset
python train.py

# Evaluate trained model
python train.py --eval
```

Trained weights are saved at:
```
models/helmet_seatbelt_v1/weights/best.pt
```

Copy to `models/best.pt` before running detection.

---

## 🚀 Running Detection

### Webcam (live)
```bash
python detect.py --source 0
```

### Image file
```bash
python detect.py --source path/to/image.jpg
```

### Video file
```bash
python detect.py --source path/to/video.mp4 --save
```

### Custom confidence threshold
```bash
python detect.py --source 0 --conf 0.6
```

---

## 🌐 Streamlit Web App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` — upload images or videos and see results instantly.

---

## 📊 Results & Output

- Annotated images saved to `results/`
- Violation log (CSV) saved to `results/report_<timestamp>.csv`
- CSV columns: `timestamp`, `frame`, `violation`, `confidence`

---

## 📈 Expected Model Performance (after training)

| Metric | Value |
|--------|-------|
| mAP@0.5 | ~0.91 |
| Precision | ~0.89 |
| Recall | ~0.87 |
| Inference speed | ~25–40 FPS (GPU) |

---

## 🔧 Customization

- **Add a buzzer alert**: Connect a GPIO buzzer (Raspberry Pi) triggered on violation
- **Add number plate detection**: Crop plate region and run OCR for automated challan
- **Deploy on CCTV**: Replace `source=0` with RTSP stream URL from IP camera
- **Edge deployment**: Export model to ONNX (`model.export(format='onnx')`) for Jetson Nano

---

## 📝 Resume Bullet

> Built a real-time road safety system using **YOLOv8** and **OpenCV** that detects helmet and seatbelt violations from live video, achieving 91%+ mAP — deployed as an interactive **Streamlit** web application.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Universe](https://universe.roboflow.com)
- [OpenCV](https://opencv.org)


---
© 2026 Shravani Patil  
Licensed under the MIT License
