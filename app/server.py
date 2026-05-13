from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

from app.predict import predict_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# open HTML dashboard
@app.get("/")
def dashboard():
    return FileResponse("ai_medical_dashboard.html")


# prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    disease, confidence, preds = predict_image(image)

    return {
        "class": disease,
        "confidence": float(confidence),
        "probabilities": preds.tolist()
    }