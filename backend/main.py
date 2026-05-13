from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import Request
from PIL import Image
import io, sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predict import predict_image
from gradcam import make_gradcam, overlay_heatmap, detect_infection_area
from chatbot import medical_chatbot
from report_pdf import create_pdf_report
from predict import get_model

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    disease, confidence, preds = predict_image(img)
    return {
        "disease": disease,
        "confidence": confidence,
        "preds": preds.tolist()
    }

@app.post("/chat")
async def chat(data: dict):
    response = medical_chatbot(data.get("message", ""), data.get("disease"))
    return {"response": response}

@app.post("/report")
async def report(data: dict):
    # Reconstruct image from base64 if needed, or store last image in session
    path = create_pdf_report(
        data["disease"], data["confidence"],
        data["severity"], data["findings"],
        data["recommendation"], None  # pass image if available
    )
    return FileResponse(path, media_type="application/pdf", filename="medical_report.pdf")