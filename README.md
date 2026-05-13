# 🩺 AI Medical Diagnosis System

An intelligent web application that analyzes **chest X-ray images** to detect lung diseases — **Normal**, **Pneumonia**, and **Tuberculosis** — using deep learning and GradCAM explainability.

---

## 🖥️ Dashboard Preview

> Upload a chest X-ray → Get instant AI diagnosis → View heatmaps → Download PDF report

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **AI Diagnosis** | Deep learning model classifies X-rays into Normal / Pneumonia / Tuberculosis |
| 🧠 **GradCAM Heatmap** | Visual explanation of which lung regions the AI focused on |
| 📊 **Clinical Analysis** | Confidence scores, risk assessment, severity levels |
| 🛡 **Precaution Plan** | Personalized care guidance based on diagnosis result |
| 📄 **PDF Report** | Downloadable professional medical report |
| 💬 **AI Doctor Chatbot** | Claude-powered assistant to answer medical questions |
| 📈 **Model Performance** | Accuracy, Precision, Recall, F1 Score dashboard |

---

## 🧰 Tech Stack

**Backend**
- Python 3.11
- FastAPI + Uvicorn
- TensorFlow / Keras
- OpenCV (GradCAM)
- ReportLab (PDF generation)
- Anthropic Claude API (chatbot)

**Frontend**
- Vanilla HTML / CSS / JavaScript
- Chart.js (confidence graphs)
- Single-file dashboard (no framework needed)

---

## 📁 Project Structure

```
medical_image_ai/
├── ai_medical_dashboard.html    # Frontend dashboard
├── requirements.txt
├── README.md
├── .gitignore
│
├── app/
│   ├── main.py                  # FastAPI server + all endpoints
│   ├── predict.py               # Model loading + prediction
│   ├── gradcam.py               # GradCAM heatmap generation
│   ├── chatbot.py               # AI Doctor (Claude API + rule-based fallback)
│   └── report_pdf.py            # PDF report generator (ReportLab)
│
├── model/                       # ⚠ Not included in repo (see below)
│   └── xray_model.h5
│
└── train/
    └── train_xray.py            # Model training script
```

---

## ⚙️ Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/medical-image-ai.git
cd medical-image-ai
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

### 4. Add your model file
Download or train the model and place it at:
```
model/xray_model.h5
```
> The model is not included in this repo due to file size. See [Model Training](#-model-training) below.

### 5. (Optional) Set Claude API key for AI chatbot
```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-your-key-here

# Mac / Linux
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```
> Without the key, the chatbot falls back to rule-based responses automatically.

### 6. Run the server
```bash
cd app
python main.py
```
The browser opens automatically at `http://localhost:8000` once the server is ready.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the dashboard HTML |
| `POST` | `/api/predict` | Run X-ray disease classification |
| `POST` | `/api/gradcam` | Generate GradCAM heatmap overlay |
| `POST` | `/api/chat` | AI Doctor chatbot response |
| `POST` | `/api/report/pdf` | Generate & download PDF report |

Full interactive API docs available at: `http://localhost:8000/docs`

---

## 🎯 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 94.2% |
| Precision | 92.0% |
| Recall | 91.0% |
| F1 Score | 91.5% |

Trained on chest X-ray dataset with 3 classes: **Normal**, **Pneumonia**, **Tuberculosis**

---

## 🏋️ Model Training

To train the model from scratch:
```bash
python train/train_xray.py
```
The trained model will be saved to `model/xray_model.h5`.

> Dataset: [Chest X-ray Images (Kaggle)](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images)

---

## 🚀 Deployment

This project is designed to run on **Render** (recommended):
- Supports persistent Python servers
- Free tier available
- Model files stored via Render Disk

See [Deployment Guide](docs/deploy.md) *(coming soon)*

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**.
It is **not** a substitute for professional medical diagnosis.
Always consult a licensed healthcare professional for medical decisions.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙌 Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Anthropic Claude](https://www.anthropic.com/)
- [Chart.js](https://www.chartjs.org/)
- [ReportLab](https://www.reportlab.com/)
