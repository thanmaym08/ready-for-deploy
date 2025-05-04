import os
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import uvicorn

app = FastAPI()

# CORS middleware (for frontend to communicate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Replace "*" with your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (optional)
if os.path.exists("static"):
    # Mount static folder to serve frontend files (HTML, CSS, JS)
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
else:
    @app.get("/")
    async def root():
        return {"message": "Static folder not found. API is running."}

# Emotion classifier model using EfficientNet
class EmotionEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EmotionEfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b4")
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the model
num_classes = 7
model = EmotionEfficientNet(num_classes)

model_path = "efficientnet_b4_emotion.pth"
if os.path.exists(model_path):
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print("❌ Error loading model:", e)
else:
    print("⚠️ Model file not found:", model_path)

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Emotion class labels
class_labels = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprised"
]

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            label = class_labels[pred_idx] if pred_idx < len(class_labels) else "unknown"

        return {
            "prediction": pred_idx,
            "label": label
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Run the app (host must be 0.0.0.0 for deployment like Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
