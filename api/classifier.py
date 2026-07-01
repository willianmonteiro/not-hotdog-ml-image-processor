"""Model loading and inference for the API.

Loads the exported Keras model once at import time (loading is slow, so we never do it
per request) and exposes a single `classify` function that turns raw image bytes into a
label + confidence.
"""

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)  # must match training

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "hotdog_classifier.h5"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run model/export.py first.")

_model = load_model(str(MODEL_PATH))


def _preprocess(image_bytes: bytes) -> np.ndarray:
    # same pipeline as training: RGB, resize, MobileNetV2 preprocess ([-1,1]), add batch dim.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    array = preprocess_input(np.asarray(image, dtype="float32"))
    return np.expand_dims(array, axis=0)


def classify(image_bytes: bytes) -> dict:
    probability = float(_model.predict(_preprocess(image_bytes))[0][0])
    is_hotdog = probability >= 0.5

    return {
        "label": "Hotdog 🌭" if is_hotdog else "Not Hotdog ❌",
        "is_hotdog": is_hotdog,
        "confidence": round(probability if is_hotdog else 1 - probability, 4),
    }
