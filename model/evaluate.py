"""Evaluate the trained CNN.

Loads the best checkpoint, reports accuracy/loss on the validation set, plots the
learning curves from training history, and builds a confusion matrix so we can see
*which* class the model gets wrong (not just the overall score).

    python model/evaluate.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from preprocessing import build_validation_generator

MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.h5"
HISTORY_PATH = CHECKPOINT_DIR / "history.json"
CLASS_NAMES = ["not hotdog", "hotdog"]  # index 0 = no, index 1 = yes


def plot_learning_curves(history: dict) -> None:
    """Plot train vs validation loss and accuracy side by side.

    The gap between the two lines is the tell-tale sign of overfitting: if training keeps
    improving while validation flattens or worsens, the model is memorizing.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(history["loss"], label="train")
    ax_loss.plot(history["val_loss"], label="validation")
    ax_loss.set(title="Loss", xlabel="epoch", ylabel="loss")
    ax_loss.legend()

    ax_acc.plot(history["accuracy"], label="train")
    ax_acc.plot(history["val_accuracy"], label="validation")
    ax_acc.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax_acc.legend()

    out = CHECKPOINT_DIR / "learning_curves.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"📈 Learning curves saved to {out}")


def plot_confusion_matrix(y_true, y_pred) -> None:
    """A confusion matrix shows counts of correct vs. each type of mistake:
    false positives (called hotdog, wasn't) and false negatives (missed a real hotdog).
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d")

    out = CHECKPOINT_DIR / "confusion_matrix.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"🔢 Confusion matrix saved to {out}")


def evaluate() -> None:
    if not BEST_MODEL_PATH.exists():
        raise SystemExit(f"❌ No model at {BEST_MODEL_PATH}. Run train.py first.")

    model = load_model(str(BEST_MODEL_PATH))
    val_gen = build_validation_generator()

    # Overall metrics.
    loss, accuracy = model.evaluate(val_gen)
    print(f"\nValidation loss:     {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")

    # per-image predictions for the confusion matrix / report. val_gen has shuffle=False,
    # so predictions line up with val_gen.classes.
    probabilities = model.predict(val_gen)
    y_pred = (probabilities > 0.5).astype(int).flatten()  # sigmoid > 0.5 -> hotdog
    y_true = val_gen.classes

    print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    plot_confusion_matrix(y_true, y_pred)

    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            plot_learning_curves(json.load(f))
    else:
        print("ℹ️  No history.json found — skipping learning curves.")


if __name__ == "__main__":
    evaluate()
