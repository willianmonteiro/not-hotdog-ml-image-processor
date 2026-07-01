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

## Dataset

The model trains on the Kaggle **"Hot Dog - Not Hot Dog"** dataset. Images aren't committed
to git — you download them yourself. Full instructions (Kaggle CLI + manual) are in
[`model/data/README.md`](model/data/README.md). Quick version:

```bash
kaggle datasets download -d dansbecker/hot-dog-not-hot-dog -p model/data --unzip
python model/prepare_data.py --source model/data/seefood   # organize into yes/no layout
python model/prepare_data.py --verify                      # sanity-check the counts
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

---

## Build steps

- [x] **1. Project structure + README skeleton**
- [x] **2. Data pipeline** — folder structure + Kaggle dataset download instructions
- [x] **3. Image preprocessing** — resize to 224×224, normalize, data augmentation
- [x] **4. CNN model architecture** — Conv2D, MaxPooling2D, Dense, binary crossentropy
- [x] **5. Training script** — early stopping + model checkpoint callbacks
- [ ] **6. Evaluation** — accuracy, loss curves, confusion matrix
- [ ] **7. Save model** for deployment (`.h5`)
- [ ] **8. FastAPI backend** — `POST /classify` endpoint
- [ ] **9. Rate limiting** with slowapi — 10 requests/min/IP
- [ ] **10. React frontend** — drag-and-drop upload + confidence display
- [ ] **11. Docker + docker-compose** — containerize everything
- [ ] **12. Final README** — full documentation, architecture diagram, design decisions

---

## License

For educational purposes. The dataset is subject to its own license on Kaggle.
