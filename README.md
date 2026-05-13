# 🩺 AI Medical Image Diagnosis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=for-the-badge&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange?style=for-the-badge&logo=tensorflow)
![Render](https://img.shields.io/badge/Deployed-Render-purple?style=for-the-badge&logo=render)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

### 🌐 **Live Demo → [ai-medical-image-diagnosis-system.onrender.com](https://ai-medical-image-diagnosis-system.onrender.com)**

*An intelligent web application that analyzes chest X-ray images to detect lung diseases using Deep Learning and GradCAM explainability.*

</div>

---

## 📸 Preview

> Upload a chest X-ray → AI Diagnosis → GradCAM Heatmap → PDF Report → AI Doctor Chat

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **AI Diagnosis** | Classifies chest X-rays into Normal / Pneumonia / Tuberculosis |
| 🧠 **GradCAM Heatmap** | Visual explanation of which lung regions the AI focused on |
| 📊 **Clinical Analysis** | Confidence scores, risk assessment and severity levels |
| 🛡 **Precaution Plan** | Personalized care guidance based on diagnosis |
| 📄 **PDF Report** | Downloadable professional medical report via ReportLab |
| 💬 **AI Doctor Chatbot** | Medical assistant to answer health questions |
| 📈 **Performance Dashboard** | Model accuracy, precision, recall and F1 metrics |

---

## 🎯 Model Performance

| Metric | Score |
|---|---|
| ✅ Accuracy | **94.2%** |
| ✅ Precision | **92.0%** |
| ✅ Recall | **91.0%** |
| ✅ F1 Score | **91.5%** |

> Trained on chest X-ray dataset with 3 classes: **Normal**, **Pneumonia**, **Tuberculosis**

---

## 🧰 Tech Stack

**Backend**
- 🐍 Python 3.11
- ⚡ FastAPI + Uvicorn
- 🤖 TensorFlow / Keras (CNN model)
- 🔬 OpenCV (GradCAM heatmap)
- 📄 ReportLab (PDF generation)

**Frontend**
- 🌐 Vanilla HTML / CSS / JavaScript
- 📊 Chart.js (confidence graphs)
- 🎨 Single-file dark theme dashboard

**Infrastructure**
- ☁️ Render (deployment)
- 🤗 HuggingFace (model hosting)
- 🐙 GitHub (version control)

---

## 📁 Project Structure

```
AI-Medical-Image-Diagnosis-System/
│
├── 🌐 ai_medical_dashboard.html     # Frontend dashboard (single file)
├── 📋 requirements.txt              # Python dependencies
├── ⚙️ render.yaml                   # Render deployment config
├── 📖 README.md
├── 🚫 .gitignore
│
├── app/
│   ├── main.py                      # FastAPI server + all endpoints
│   ├── predict.py                   # Model loading + auto-download + prediction
│   ├── gradcam.py                   # GradCAM heatmap generation
│   ├── chatbot.py                   # AI Doctor (rule-based medical assistant)
│   └── report_pdf.py                # PDF report generator
│
├── model/                           # ⚠ Not in repo — hosted on HuggingFace
│   └── xray_model.h5
│
└── train/
    └── train_xray.py                # Model training script
```

---

## ⚙️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Rabbani34/AI-Medical-Image-Diagnosis-System.git
cd AI-Medical-Image-Diagnosis-System
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model file
Download the model from HuggingFace and place it at:
```
model/xray_model.h5
```
🤗 Model: [Reshal7/xray-diagnosis-model](https://huggingface.co/Reshal7/xray-diagnosis-model)

### 5. Run the server
```bash
cd app
python main.py
```
✅ Browser opens automatically at `http://localhost:8000`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the dashboard |
| `POST` | `/api/predict` | X-ray disease classification |
| `POST` | `/api/gradcam` | GradCAM heatmap generation |
| `POST` | `/api/chat` | AI Doctor chatbot |
| `POST` | `/api/report/pdf` | PDF report download |

📚 Interactive API docs: [`/docs`](https://ai-medical-image-diagnosis-system.onrender.com/docs)

---

## ☁️ Deployment

Deployed on **Render** with model hosted on **HuggingFace**.

On startup, the server automatically downloads the model from HuggingFace if not found locally — no manual setup needed on the server.

### Environment Variables (Render)

| Variable | Description |
|---|---|
| `MODEL_URL` | HuggingFace model download URL |
| `ANTHROPIC_API_KEY` | Optional — enables Claude AI chatbot |

---

## 🚀 Roadmap

- [x] Chest X-ray diagnosis (Normal / Pneumonia / TB)
- [x] GradCAM heatmap visualization
- [x] PDF report generation
- [x] AI Doctor chatbot
- [x] Deployed on Render
- [ ] 🧠 MRI brain scan analysis
- [ ] 👁 Retina disease detection
- [ ] 🦠 Skin disease classification
- [ ] 📱 Mobile responsive design
- [ ] 🌍 Multi-language support

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**.
It is **not** a substitute for professional medical diagnosis.
Always consult a licensed healthcare professional for medical decisions.

---

## 📄 License

MIT License — feel free to use, modify, and distribute with attribution.

---

## 🙌 Acknowledgements

- [TensorFlow](https://www.tensorflow.org/) — Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Chart.js](https://www.chartjs.org/) — Beautiful charts
- [ReportLab](https://www.reportlab.com/) — PDF generation
- [HuggingFace](https://huggingface.co/) — Model hosting
- [Render](https://render.com/) — Cloud deployment

---

<div align="center">

**Made with ❤️ for better healthcare accessibility**

⭐ Star this repo if you found it useful!

</div>
