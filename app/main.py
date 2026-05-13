"""
AI Medical Diagnosis System — FastAPI Backend
=============================================
Endpoints
---------
GET  /                  → serves ai_medical_dashboard.html
POST /api/predict       → disease prediction
POST /api/gradcam       → GradCAM heatmap + infection metrics
POST /api/chat          → AI doctor chatbot
POST /api/report/pdf    → stream PDF medical report
"""

from __future__ import annotations
import io, base64, os, numpy as np
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from predict    import predict_image, get_model
from gradcam    import make_gradcam, overlay_heatmap, detect_infection_area
from chatbot    import medical_chatbot
from report_pdf import create_pdf_report

app = FastAPI(title="AI Medical Diagnosis API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent


# ── Pre-load model at startup (avoids slow first request) ─────────────────────
@app.on_event("startup")
async def _preload():
    print("🔄 Pre-loading AI model…")
    get_model()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _open_image(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(upload.file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

def _np_to_b64(arr: np.ndarray, fmt="JPEG") -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format=fmt)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


# ══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    f = BASE_DIR / "ai_medical_dashboard.html"
    if not f.exists():
        raise HTTPException(404, "Dashboard HTML not found.")
    return HTMLResponse(content=f.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    disease, confidence, preds = predict_image(_open_image(file))
    if disease is None:
        raise HTTPException(500, "Model prediction failed. Check MODEL_URL env var.")
    return JSONResponse({
        "disease":    disease,
        "confidence": round(float(confidence), 2),
        "preds":      [round(float(p), 6) for p in preds],
    })


# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/gradcam")
async def api_gradcam(file: UploadFile = File(...), intensity: float = Form(0.4)):
    image = _open_image(file)
    model = get_model()
    if model is None:
        raise HTTPException(500, "Model not loaded.")

    heatmap = make_gradcam(image, model)
    overlay = overlay_heatmap(image, heatmap, alpha=max(0.1, min(1.0, intensity)))
    boxes, pct = detect_infection_area(heatmap)

    region   = "Minimal" if pct < 10 else "Localized" if pct < 30 else "Significant"
    severity = "Very low probability" if pct < 10 else "Moderate probability" if pct < 30 else "High probability"

    return JSONResponse({
        "overlay_b64":       _np_to_b64(overlay),
        "infection_percent": round(float(pct), 2),
        "region_label":      region,
        "severity_label":    severity,
        "boxes":             [[int(x),int(y),int(w),int(h)] for x,y,w,h in boxes],
    })


# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/chat")
async def api_chat(message: str = Form(...), disease: str = Form("")):
    return JSONResponse({"response": medical_chatbot(message, disease.strip().upper() or None)})


# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/report/pdf")
async def api_report_pdf(
    file: UploadFile = File(...),
    disease: str = Form(...), confidence: float = Form(...),
    severity: str = Form(...), findings: str = Form(...),
    recommendation: str = Form(...),
):
    pdf_path = create_pdf_report(
        disease=disease.upper(), confidence=float(confidence),
        severity=severity, findings=findings,
        recommendation=recommendation, image=_open_image(file),
    )
    if not os.path.exists(pdf_path):
        raise HTTPException(500, "PDF generation failed.")

    def _stream():
        with open(pdf_path, "rb") as f:
            yield from f
        try: os.remove(pdf_path)
        except OSError: pass

    return StreamingResponse(_stream(), media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="medical_report.pdf"'})


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL DEV ONLY — Render uses startCommand from render.yaml
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn, webbrowser, threading, time, urllib.request

    def _open_browser():
        url = "http://localhost:8000"
        print("\n⏳  Waiting for server to be ready...")
        for _ in range(60):
            try:
                urllib.request.urlopen(url, timeout=1); break
            except: time.sleep(1)
        webbrowser.open(url)
        print(f"✅  Opened {url} — Press CTRL+C to stop.\n")

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
