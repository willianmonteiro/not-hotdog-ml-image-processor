# 🌭 Not Hotdog — ML Image Processor

> Inspired by the "SeeFood" app from HBO's *Silicon Valley*. Upload an image and the app
> tells you whether it's a **Hotdog 🌭** or **Not Hotdog ❌**, with a confidence score.

A full-stack machine learning project built to be **readable and educational** — every
component is documented.

---

## What it does

1. You upload or drag-and-drop an image in the browser.
2. A FastAPI backend runs it through a trained **Convolutional Neural Network (CNN)**.
3. You get back a label (`Hotdog` / `Not Hotdog`) and a confidence percentage.

**Stack:** TensorFlow/Keras (model) · FastAPI (API) · React + TypeScript (frontend) · Docker.

---

## Getting started

**Prerequisites:** Python 3.9–3.12 and a free [Kaggle](https://www.kaggle.com) account (for the dataset).

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r model/requirements.txt          # training/eval deps (TensorFlow, etc.)

# On Python 3.12, TensorFlow ships Keras 3 (no ImageDataGenerator). Force the Keras 2 API:
pip install tf-keras
export TF_USE_LEGACY_KERAS=1                    # keep this set for all model commands below
```

### 2. Get the dataset

Images aren't committed to git — you download them from Kaggle. Full instructions (CLI +
manual) are in [`model/data/README.md`](model/data/README.md). Quick version:

```bash
pip install kaggle                             # then add your token at ~/.kaggle/kaggle.json

# Download into an isolated raw/ folder so it never collides with train/ and validation/.
kaggle datasets download -d dansbecker/hot-dog-not-hot-dog -p model/data/raw --unzip
python model/prepare_data.py --source model/data/raw/seefood --clean   # organize + remove raw
python model/prepare_data.py --verify                                  # sanity-check the counts
```

### 3. Train, evaluate and export the model

```bash
python model/train.py       # trains the CNN -> model/checkpoints/best_model.h5
python model/evaluate.py    # accuracy + learning curves + confusion matrix (optional)
python model/export.py      # -> model/hotdog_classifier.h5 (the artifact the API loads)
```

Quick sanity check without the API:

```bash
python model/predict.py path/to/image.jpg      # => Hotdog 🌭  (98.2% confident)
```

### 4. Run the API

```bash
pip install -r api/requirements.txt
uvicorn main:app --reload --app-dir api
```

Open <http://localhost:8000/docs> for interactive docs, or call it directly:

```bash
curl -F "file=@path/to/image.jpg" http://localhost:8000/classify
# {"label": "Hotdog 🌭", "is_hotdog": true, "confidence": 0.98}
```

---

## How the model works

The classifier is a small **Convolutional Neural Network (CNN)** — the standard architecture
for image tasks. The key concepts:

- **Conv2D** — slides small filters across the image to detect visual patterns. Early layers
  pick up edges and colors; deeper layers combine those into textures and shapes. Filters
  double each block (32 → 64 → 128) as the network learns richer features.
- **MaxPooling2D** — downsamples by keeping the strongest value in each 2×2 region. It shrinks
  the data (less compute) and makes detection robust to small shifts in position.
- **Flatten → Dense** — turns the final feature maps into a flat vector and feeds a fully
  connected layer that reasons over the combined features.
- **Dropout (0.5)** — randomly disables half the neurons during training so the model can't
  lean on any single one; a simple, effective guard against overfitting.
- **Sigmoid output** — one neuron producing a probability from 0 to 1 (the hotdog confidence).
- **Binary crossentropy** — the loss function that matches a single-probability, two-class setup.

```
224×224×3 → [Conv 32 → Pool] → [Conv 64 → Pool] → [Conv 128 → Pool] → Flatten → Dense 128 → Dropout → Sigmoid
```

### Training

`python model/train.py` fits the model and uses two callbacks:

- **EarlyStopping** — stops training once validation loss stops improving (patience 5) and
  restores the best weights, so we don't waste epochs or keep an overfit final model.
- **ModelCheckpoint** — saves the model every time validation loss hits a new best, so the
  best result survives a crash or later overfitting.

The best model is written to `model/checkpoints/best_model.h5` and per-epoch metrics to
`history.json` (used for the learning curves in evaluation).

### Evaluation

`python model/evaluate.py` loads the best checkpoint and reports how well it generalizes:

- **Accuracy / loss** on the validation set (images the model never trained on).
- **Learning curves** — train vs. validation loss/accuracy over epochs. A widening gap
  between the two lines signals overfitting.
- **Confusion matrix** — breaks results into correct predictions vs. the two mistake types:
  false positives (called hotdog, wasn't) and false negatives (missed a real hotdog).

Plots are saved to `model/checkpoints/` (`learning_curves.png`, `confusion_matrix.png`).

---

## Roadmap

**Model pipeline** — `model/` &nbsp;✅
- Data prep from the Kaggle dataset (`prepare_data.py`)
- Preprocessing: resize, normalize, augmentation (`preprocessing.py`)
- CNN architecture (`model.py`) + training with callbacks (`train.py`)
- Evaluation and export to a deploy-ready `.h5` (`evaluate.py`, `export.py`, `predict.py`)

**API** — `api/` &nbsp;🚧
- ✅ FastAPI service with `POST /classify` returning label + confidence
- ⬜ Rate limiting with slowapi (10 requests/min/IP)

**Frontend** — `frontend/` &nbsp;⬜
- Drag-and-drop image upload, live result with confidence

**Infrastructure** &nbsp;⬜
- Docker + docker-compose to run API and frontend together

---

## License

For educational purposes. The dataset is subject to its own license on Kaggle.
