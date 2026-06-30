"""
prepare_data.py — Organize the Kaggle "Hot Dog - Not Hot Dog" dataset into our layout.

The Kaggle dataset ships with its own folder names (`hot_dog` / `not_hot_dog`) and its own
split names (`train` / `test`). Our training code expects a specific, predictable layout:

    model/data/train/yes,  model/data/train/no
    model/data/validation/yes,  model/data/validation/no

USAGE
-----
    python model/prepare_data.py --source model/data/seefood   # reorganize the download
    python model/prepare_data.py --verify                      # just count what's there
"""

import argparse
import shutil
from pathlib import Path

# --- Path configuration -------------------------------------------------------------------
# We resolve paths relative to THIS file (not the current working directory) so the script
# works no matter where you run it from. `__file__` is this script; its parent is `model/`.
MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_DIR / "data"

TARGET_DIRS = {
    ("train", "yes"): DATA_DIR / "train" / "yes",
    ("train", "no"): DATA_DIR / "train" / "no",
    ("validation", "yes"): DATA_DIR / "validation" / "yes",
    ("validation", "no"): DATA_DIR / "validation" / "no",
}

# Kaggle uses `train`/`test`; we call them `train`/`validation`. Kaggle uses
# `hot_dog`/`not_hot_dog`; we use `yes`/`no`.
#   (kaggle_split, kaggle_class) -> (our_split, our_class)
KAGGLE_TO_TARGET = {
    ("train", "hot_dog"): ("train", "yes"),
    ("train", "not_hot_dog"): ("train", "no"),
    ("test", "hot_dog"): ("validation", "yes"),
    ("test", "not_hot_dog"): ("validation", "no"),
}

# Guards against copying stray files (e.g. a .DS_Store).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _count_images(folder: Path) -> int:
    """Count image files directly inside `folder` (non-recursive)."""
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def reorganize(source: Path) -> None:
    """
    Copy images from Kaggle's `seefood/` structure into our train/validation yes/no layout.

    We COPY rather than MOVE on purpose: if the reorganization is wrong, the original
    download is still intact and can re-run without re-downloading from Kaggle.
    """
    if not source.exists():
        raise SystemExit(f"❌ Source folder not found: {source}\n"
                         f"   Download the dataset first (see model/data/README.md).")

    total_copied = 0
    for (kaggle_split, kaggle_class), (our_split, our_class) in KAGGLE_TO_TARGET.items():
        src_folder = source / kaggle_split / kaggle_class
        dst_folder = TARGET_DIRS[(our_split, our_class)]
        dst_folder.mkdir(parents=True, exist_ok=True)

        if not src_folder.exists():
            print(f"⚠️  Skipping missing source folder: {src_folder}")
            continue

        copied = 0
        for img in src_folder.iterdir():
            if img.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(img, dst_folder / img.name)  # copy2 preserves file metadata
                copied += 1

        total_copied += copied
        print(f"  {kaggle_split}/{kaggle_class:12s} → {our_split}/{our_class:3s}  "
              f"({copied} images)")

    print(f"\n✅ Copied {total_copied} images into the train/validation layout.")
    verify()


def verify() -> None:
    """Print how many images are in each class folder, and flag class imbalance.

    If one class has far more images than the other, the model can hit
    high accuracy by always guessing the majority class while learning nothing useful. A
    roughly balanced dataset forces it to actually learn the difference. This dataset is
    balanced by design, but verifying keeps us honest if files go missing.
    """
    print("\nDataset summary")
    print("-" * 40)
    print(f"{'split':<12}{'yes (hotdog)':>14}{'no (not)':>12}")
    print("-" * 40)

    for split in ("train", "validation"):
        n_yes = _count_images(TARGET_DIRS[(split, "yes")])
        n_no = _count_images(TARGET_DIRS[(split, "no")])
        print(f"{split:<12}{n_yes:>14}{n_no:>12}")

        # Warn on heavy imbalance (one class > 1.5x the other), but only if both are present.
        if n_yes and n_no:
            ratio = max(n_yes, n_no) / min(n_yes, n_no)
            if ratio > 1.5:
                print(f"  ⚠️  '{split}' is imbalanced ({ratio:.1f}x). "
                      f"Consider balancing the classes.")
    print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path,
                        help="Path to the unzipped Kaggle 'seefood' folder.")
    parser.add_argument("--verify", action="store_true",
                        help="Only count images in the current layout (no copying).")
    args = parser.parse_args()

    if args.verify:
        verify()
    elif args.source:
        reorganize(args.source)
    else:
        parser.error("Provide --source <path> to organize the data, or --verify to count it.")


if __name__ == "__main__":
    main()
