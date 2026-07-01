"""Classify a single image from the command line.

A quick way to sanity-check the exported model without spinning up the API.

    python model/predict.py path/to/image.jpg
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_DIR = Path(__file__).resolve().parent
DEPLOY_MODEL_PATH = MODEL_DIR / "hotdog_classifier.h5"
IMG_SIZE = (224, 224)  # must match training


def preprocess(image_path: Path) -> np.ndarray:
    # same steps as training: RGB, resize to 224x224, scale to 0-1, add a batch dimension.
    image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(array, axis=0)


def predict(image_path: Path) -> None:
    if not DEPLOY_MODEL_PATH.exists():
        raise SystemExit(f"❌ No model at {DEPLOY_MODEL_PATH}. Run export.py first.")

    model = load_model(str(DEPLOY_MODEL_PATH))
    probability = float(model.predict(preprocess(image_path))[0][0])

    is_hotdog = probability >= 0.5
    label = "Hotdog 🌭" if is_hotdog else "Not Hotdog ❌"
    confidence = probability if is_hotdog else 1 - probability

    print(f"{label}  ({confidence:.1%} confident)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Path to the image to classify.")
    predict(parser.parse_args().image)
