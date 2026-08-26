"""Model serving app (FastAPI).

Endpoints:
  GET  /health  -> 200 with {"status": "ok", "model_loaded": true} once ready.
  POST /predict -> accepts an uploaded image (multipart field "image"),
                   returns predicted class + per-class probabilities.

The checkpoint path is configurable via CHECKPOINT_PATH (default
/app/checkpoints/classifier_v1.pt). The model is loaded once at startup.
"""
from __future__ import annotations

import io
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from dataset import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES
from model import get_model

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")

app = FastAPI(title="CIFAR-10 Classifier", version="1.0")

_state: dict = {"model": None, "device": None}

_infer_tf = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


def load_model() -> None:
    """Load the checkpoint into memory. Safe to call once at startup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = get_model(
        architecture=ckpt.get("architecture", "resnet18"),
        num_classes=ckpt.get("num_classes", len(CLASS_NAMES)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    _state["model"], _state["device"] = model, device


@app.on_event("startup")
def _startup() -> None:
    # Don't crash the process if the checkpoint isn't present yet; /health will
    # report not-ready until it can be loaded (useful during rollout).
    try:
        load_model()
    except Exception as exc:  # noqa: BLE001
        print(f"[serve] model not loaded at startup: {exc}", flush=True)


@app.get("/health")
def health():
    if _state["model"] is None:
        # try a lazy load in case the checkpoint appeared after startup
        try:
            load_model()
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="invalid image file")

    x = _infer_tf(img).unsqueeze(0).to(_state["device"])
    with torch.no_grad():
        probs = F.softmax(_state["model"](x), dim=1).squeeze(0).cpu().tolist()

    ranked = sorted(zip(CLASS_NAMES, probs), key=lambda t: t[1], reverse=True)
    return {
        "predicted_class": ranked[0][0],
        "confidence": round(ranked[0][1], 4),
        "probabilities": {c: round(p, 4) for c, p in zip(CLASS_NAMES, probs)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
