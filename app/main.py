"""
AI Medical Diagnosis System — FastAPI Backend
=============================================
Run:  uvicorn main:app --reload --port 8000
Then: open http://localhost:8000

Endpoints
---------
GET  /                  → serves ai_medical_dashboard.html
POST /api/predict       → disease prediction
POST /api/gradcam       → GradCAM heatmap + infection metrics
POST /api/chat          → AI doctor chatbot
POST /api/report/pdf    → stream PDF medical report
"""

from __future__ import annotations
import io
import base64
import os
import numpy as np
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from predict          import predict_image, get_model
from gradcam          import make_gradcam, overlay_heatmap, detect_infection_area
from chatbot          import medical_chatbot
from report_pdf       import create_pdf_report

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Medical Diagnosis API", version="2.0.0")

# Allow the HTML page (same origin) and any local dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent   # project root


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _open_image(upload: UploadFile) -> Image.Image:
    raw = upload.file.read()
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")


def _np_to_b64(arr: np.ndarray, fmt: str = "JPEG") -> str:
    """Convert HWC uint8 numpy array → data-URI base64 string."""
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


# ══════════════════════════════════════════════════════════════════════════════
# SERVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_file = BASE_DIR / "ai_medical_dashboard.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard HTML not found.")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/predict
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    """
    Run the X-ray disease classifier.

    Response
    --------
    {
        "disease":    "NORMAL" | "PNEUMONIA" | "TUBERCULOSIS",
        "confidence": float  (0-100),
        "preds":      [normal_prob, pneumonia_prob, tb_prob]  (0-1 each)
    }
    """
    image = _open_image(file)
    disease, confidence, preds = predict_image(image)

    if disease is None:
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed. Verify MODEL_PATH in predict.py.",
        )

    return JSONResponse({
        "disease":    disease,
        "confidence": round(float(confidence), 2),
        "preds":      [round(float(p), 6) for p in preds],
    })


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/gradcam
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/gradcam")
async def api_gradcam(
    file:      UploadFile = File(...),
    intensity: float      = Form(0.4),
):
    """
    Generate GradCAM heatmap overlay + infection bounding boxes.

    Response
    --------
    {
        "overlay_b64":       "data:image/jpeg;base64,…",
        "infection_percent": float,
        "region_label":      "Minimal" | "Localized" | "Significant",
        "severity_label":    str,
        "boxes":             [[x, y, w, h], …]
    }
    """
    image = _open_image(file)
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    alpha   = max(0.1, min(1.0, float(intensity)))
    heatmap = make_gradcam(image, model)
    overlay = overlay_heatmap(image, heatmap, alpha=alpha)
    boxes, infection_percent = detect_infection_area(heatmap)

    if infection_percent < 10:
        region   = "Minimal"
        severity = "Very low probability of infection"
    elif infection_percent < 30:
        region   = "Localized"
        severity = "Moderate probability of infection"
    else:
        region   = "Significant"
        severity = "High probability of infection"

    return JSONResponse({
        "overlay_b64":       _np_to_b64(overlay),
        "infection_percent": round(float(infection_percent), 2),
        "region_label":      region,
        "severity_label":    severity,
        "boxes":             [[int(x), int(y), int(w), int(h)] for x, y, w, h in boxes],
    })


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/chat
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def api_chat(
    message: str = Form(...),
    disease: str = Form(""),
):
    """
    AI Doctor chatbot.

    Body
    ----
    message : str  — user's question
    disease : str  — "NORMAL" | "PNEUMONIA" | "TUBERCULOSIS" | "" (optional)

    Response
    --------
    { "response": str }
    """
    disease_clean = disease.strip().upper() or None
    response = medical_chatbot(message, disease_clean)
    return JSONResponse({"response": response})


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/report/pdf
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/report/pdf")
async def api_report_pdf(
    file:           UploadFile = File(...),
    disease:        str        = Form(...),
    confidence:     float      = Form(...),
    severity:       str        = Form(...),
    findings:       str        = Form(...),
    recommendation: str        = Form(...),
    preds:          str        = Form(""),   # JSON array string, optional
):
    """
    Generate and stream a PDF medical report.

    Body (multipart/form-data)
    --------------------------
    file           : UploadFile  — the X-ray image
    disease        : str
    confidence     : float (0-100)
    severity       : str
    findings       : str
    recommendation : str
    preds          : str  — JSON "[0.02, 0.91, 0.07]" (optional)
    """
    import json

    image = _open_image(file)

    preds_list = None
    if preds.strip():
        try:
            preds_list = json.loads(preds)
        except Exception:
            pass

    pdf_path = create_pdf_report(
        disease        = disease.upper(),
        confidence     = float(confidence),
        severity       = severity,
        findings       = findings,
        recommendation = recommendation,
        image          = image,
        preds          = preds_list,
    )

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="PDF generation failed.")

    def _stream():
        with open(pdf_path, "rb") as f:
            yield from f
        try:
            os.remove(pdf_path)
        except OSError:
            pass

    return StreamingResponse(
        _stream(),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="medical_report.pdf"'},
    )


# ══════════════════════════════════════════════════════════════════════════════
# DEV RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time
    import urllib.request

    def _open_browser():
        url = "http://localhost:8000"
        print("\n⏳  Waiting for server to be ready...")
        # Poll every second until server actually responds (up to 60s)
        for _ in range(60):
            try:
                urllib.request.urlopen(url, timeout=1)
                break          # server is up
            except Exception:
                time.sleep(1)
        webbrowser.open(url)
        print(f"✅  Opened {url} in your browser.")
        print("   Press CTRL+C to stop the server.\n")

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)