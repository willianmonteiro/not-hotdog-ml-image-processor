"""Train the hotdog / not-hotdog CNN.

Loads the data generators, builds the model, and fits it with two callbacks:
EarlyStopping (halt when validation stops improving) and ModelCheckpoint (keep the
best weights, not just the last epoch). Saves the best model and the training history.

    python model/train.py
"""

import json
from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import build_model
from preprocessing import create_data_generators

MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.h5"
HISTORY_PATH = CHECKPOINT_DIR / "history.json"

# upper bound on passes over the data
EPOCHS = 50


def train():
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    train_gen, val_gen = create_data_generators()
    model = build_model()

    callbacks = [
        # stop once val_loss stops improving for `patience` epochs, and roll back to the
        # best weights so we don't keep the overfit final ones
        # https://keras.io/api/callbacks/early_stopping/
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        # save the model whenever val_loss hits a new best, so a crash or later overfitting
        # can never lose our best result
        # https://keras.io/api/callbacks/model_checkpoint/
        ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # persist the metrics per epoch so evaluate.py can plot the learning curves later
    with open(HISTORY_PATH, "w") as f:
        json.dump(history.history, f, indent=2)

    print(f"\n✅ Best model saved to {BEST_MODEL_PATH}")
    print(f"   Training history saved to {HISTORY_PATH}")


if __name__ == "__main__":
    train()
