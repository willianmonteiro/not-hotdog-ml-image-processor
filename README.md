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

The classifier uses **transfer learning**: a **MobileNetV2** base pretrained on ImageNet
(1.4M images) is frozen and used as a feature extractor, with a small trainable head on top.

```
224×224×3 → MobileNetV2 (frozen, pretrained) → GlobalAveragePooling → Dropout → Sigmoid
```

- **MobileNetV2 (frozen base)** — already knows how to see edges, textures and shapes from
  millions of images. We keep those weights fixed and reuse them.
- **GlobalAveragePooling2D** — collapses each feature map to a single number, giving a compact
  1280-length vector (far fewer parameters than flattening, so less overfitting).
- **Dropout (0.3)** — randomly disables neurons during training to reduce overfitting.
- **Sigmoid output** — one neuron producing a probability from 0 to 1 (the hotdog confidence).
- **Binary crossentropy** — the loss function that matches a single-probability, two-class setup.

Because MobileNetV2 expects its own input scaling, preprocessing uses `preprocess_input`
(pixels mapped to `[-1, 1]`) instead of a plain `/255` — applied identically at train and
inference time.

### Why transfer learning? (a real design decision)

The first version was a **CNN trained from scratch** (stacked Conv2D + MaxPooling blocks). On
this dataset it failed — after training and running evaluation it sat at exactly **50%
accuracy** (`loss ≈ 0.69`, i.e. random guessing), predicting "hotdog" for almost everything:

```
Validation accuracy: 0.5000
              precision  recall  f1-score
  not hotdog     0.50     0.04     0.08
      hotdog     0.50     0.96     0.66
```

The reason is data size: ~500 images is far too few to teach a network millions of weights
from zero. Transfer learning sidesteps this by reusing features already learned from millions
of images, so only a tiny head needs training — which is exactly what small datasets need.

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
