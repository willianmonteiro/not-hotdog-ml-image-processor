"""Export the trained checkpoint as a deploy-ready model.

Takes the best training checkpoint and re-saves it without the optimizer state — the
API only needs to run inference, so dropping the optimizer shrinks the file. The result
is the single artifact the API loads.

    python model/export.py
"""

from pathlib import Path

from tensorflow.keras.models import load_model

MODEL_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = MODEL_DIR / "checkpoints" / "best_model.h5"
DEPLOY_MODEL_PATH = MODEL_DIR / "hotdog_classifier.h5"


def export() -> None:
    if not BEST_MODEL_PATH.exists():
        raise SystemExit(f"❌ No checkpoint at {BEST_MODEL_PATH}. Run train.py first.")

    model = load_model(str(BEST_MODEL_PATH))
    model.save(str(DEPLOY_MODEL_PATH), include_optimizer=False)  # tf-keras wants str paths
    print(f"✅ Deploy model saved to {DEPLOY_MODEL_PATH}")


if __name__ == "__main__":
    export()
