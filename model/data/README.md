# Dataset — Hotdog / Not Hotdog

The model is trained on the classic **"Hot Dog - Not Hot Dog"** dataset from Kaggle
(originally created for the *Silicon Valley* "SeeFood" gag). It's a small, balanced,
two-class image dataset — perfect for learning, and small enough to train on a laptop.

> The images themselves are **not** committed to git (they're large and license-bound).
> This folder only keeps the directory structure via `.gitkeep` files. You download the
> data yourself with the steps below.

---

## Target folder layout

After setup, this folder must look exactly like this — the training code depends on it:

```
model/data/
├── train/
│   ├── yes/    # hotdog images      🌭  (class 1, the "positive" class)
│   └── no/     # not-hotdog images  ❌  (class 0, the "negative" class)
└── validation/
    ├── yes/
    └── no/
```

> **Why this naming?** Keras assigns class indices **alphabetically**, so `no` → 0 and
> `yes` → 1. That makes *hotdog* the positive class (1), so a prediction near `1.0`
> reads intuitively as "very confident it's a hotdog".

---

## Option A — Kaggle CLI (recommended, fully automated)

**1. Install and authenticate the Kaggle CLI.**

```bash
pip install kaggle
```

Create an API token at <https://www.kaggle.com/settings/account> → *"Create New Token"*.
This downloads `kaggle.json`. Place it at `~/.kaggle/kaggle.json` and lock it down:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json   # Kaggle CLI refuses to run if the token is world-readable
```

**2. Download and unzip the dataset.**

```bash
# Run from the repo root
kaggle datasets download -d dansbecker/hot-dog-not-hot-dog -p model/data --unzip
```

This gives you a `seefood/` folder with Kaggle's own structure:

```
model/data/seefood/
├── train/
│   ├── hot_dog/
│   └── not_hot_dog/
└── test/
    ├── hot_dog/
    └── not_hot_dog/
```

**3. Reorganize it into our `yes`/`no` layout.**

```bash
python model/prepare_data.py --source model/data/seefood
```

That's it — `prepare_data.py` copies the images into the right folders and prints a count.

---

## Option B — Manual download

1. Go to <https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog>.
2. Click **Download** (you'll need a free Kaggle account) and unzip it.
3. Run the same reorganize step, pointing `--source` at the unzipped `seefood` folder:

```bash
python model/prepare_data.py --source /path/to/unzipped/seefood
```

---

## Verify your setup

```bash
python model/prepare_data.py --verify
```

You should see roughly:

| Split        | yes (hotdog) | no (not hotdog) |
| ------------ | ------------ | --------------- |
| `train/`     | ~249         | ~249            |
| `validation/`| ~250         | ~250            |

The dataset is **balanced** (equal images per class) on purpose — see the note in
`prepare_data.py` for why class balance matters.
