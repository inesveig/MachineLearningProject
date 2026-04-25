"""
MNIST Data Loading & Preprocessing  –  Section 1.1
===================================================
Source: https://github.com/mbornet-hl/MNIST/tree/master/IMAGES/GROUPS

Each PNG in that repo is a *sprite sheet*: a mosaic of 1 000 handwritten
digit images (28 × 28 px) arranged in a 25-column × 40-row grid.
There is one sprite sheet per digit class (0–9), so 10 files in total.

Pipeline
--------
1. Download the 10 sprite sheets from GitHub (raw URLs).
2. Slice each sheet into 1 000 individual 28 × 28 patches.
3. Convert each patch to grayscale if needed and flatten it to ℝ^784.
4. Normalise pixel intensities from [0, 255] → [0, 1].
5. Stack everything into:
       X  –  NumPy array of shape  (10 000, 784),  dtype float32
       y  –  NumPy array of shape  (10 000,),       dtype int32
6. Shuffle the full dataset (reproducible, seeded).
7. Split into a training set (80 %) and a test set (20 %).
"""

import io
import urllib.request

import numpy as np
from PIL import Image                # Pillow  –  pip install pillow
from sklearn.model_selection import train_test_split   # pip install scikit-learn

# ── Constants ────────────────────────────────────────────────────────────────

TILE_SIZE   = 28          # each individual digit image is 28 × 28 pixels
GRID_COLS   = 25          # sprite-sheet layout: 25 columns …
GRID_ROWS   = 40          # … and 40 rows  →  25 × 40 = 1 000 images per sheet
N_PER_CLASS = GRID_COLS * GRID_ROWS   # 1 000

# The raw-file URL template.  GitHub serves raw PNGs at this base path.
RAW_BASE = (
    "https://raw.githubusercontent.com/mbornet-hl/MNIST/master/IMAGES/GROUPS/"
)

# Exact filenames as they appear in the repository (one per digit 0–9).
# Pattern:  mnist_v5_MNIST-{digit}_{start:05d}-{end:05d}_25x40.png
# The numeric range in the name reflects the original MNIST sample indices
# for that class; we only need the digit embedded after "MNIST-".
FILENAMES = [
    "mnist_v5_MNIST-0_00001-01000_25x40.png",
    "mnist_v5_MNIST-1_01001-02000_25x40.png",
    "mnist_v5_MNIST-2_01001-02000_25x40.png",
    "mnist_v5_MNIST-3_00001-01000_25x40.png",
    "mnist_v5_MNIST-4_00001-01000_25x40.png",
    "mnist_v5_MNIST-5_00001-01000_25x40.png",
    "mnist_v5_MNIST-6_00001-01000_25x40.png",
    "mnist_v5_MNIST-7_00001-01000_25x40.png",
    "mnist_v5_MNIST-8_00001-01000_25x40.png",
    "mnist_v5_MNIST-9_00001-01000_25x40.png",
]

TEST_SIZE   = 0.20    # 20 % of total data reserved for testing
RANDOM_SEED = 42      # fixed seed → reproducible split across runs


# ── Helper functions ──────────────────────────────────────────────────────────

def fetch_image(url: str) -> Image.Image:
    """Download a PNG from *url* and return a Pillow Image object."""
    with urllib.request.urlopen(url) as response:
        raw_bytes = response.read()
    return Image.open(io.BytesIO(raw_bytes))


def slice_sprite_sheet(sheet: Image.Image, digit: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cut a sprite sheet into individual 28 × 28 tiles.

    Parameters
    ----------
    sheet : PIL Image
        The full mosaic image (700 × 1120 px for a 25 × 40 grid of 28 × 28 tiles).
    digit : int
        The class label (0–9) for every tile in this sheet.

    Returns
    -------
    X_class : float32 array of shape (1 000, 784)
        Flattened, normalised pixel vectors.
    y_class : int32 array of shape (1 000,)
        Label repeated 1 000 times.
    """
    # Convert to grayscale (mode 'L') in case the PNG has an RGB or RGBA mode.
    # Grayscale is what the mathematical model expects: one intensity per pixel.

    sheet_gray = sheet.convert("L")
    sheet_arr = np.array(sheet_gray, dtype=np.float32)

    actual_h, actual_w = sheet_arr.shape

    # Recompute grid dimensions from the actual image size
    cols = actual_w // TILE_SIZE  # integer division — ignores any remainder
    rows = actual_h // TILE_SIZE

    patches = []
    for row in range(rows):
        for col in range(cols):
            y0, y1 = row * TILE_SIZE, (row + 1) * TILE_SIZE
            x0, x1 = col * TILE_SIZE, (col + 1) * TILE_SIZE
            tile = sheet_arr[y0:y1, x0:x1]  # guaranteed (28, 28)
            patches.append(tile.flatten())

    X_class = np.array(patches, dtype=np.float32)
    X_class /= 255.0

    y_class = np.full(len(patches), fill_value=digit, dtype=np.int32)

    return X_class, y_class


# ── Main loading function ─────────────────────────────────────────────────────

def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Download, preprocess, and split the MNIST sprite-sheet dataset.

    Returns
    -------
    X_train : float32 array, shape (8 000, 784)
    X_test  : float32 array, shape (2 000, 784)
    y_train : int32   array, shape (8 000,)
    y_test  : int32   array, shape (2 000,)

    Each row of X_* is a vector x_i ∈ [0,1]^784 representing one image.
    Each element of y_* is the corresponding digit label in {0, …, 9}.
    """
    all_X, all_y = [], []

    for digit, filename in enumerate(FILENAMES):
        url = RAW_BASE + filename
        print(f"  Downloading digit {digit}  ←  {filename} …", end=" ", flush=True)

        sheet = fetch_image(url)
        X_class, y_class = slice_sprite_sheet(sheet, digit)

        all_X.append(X_class)
        all_y.append(y_class)
        print(f"done  ({X_class.shape[0]} images)")

    # Stack all classes into one dataset
    X = np.vstack(all_X)   # (10 000, 784)
    y = np.concatenate(all_y)       # (10 000,)

    # Shuffle and split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,          # preserve class balance in both sets
    )

    return X_train, X_test, y_train, y_test


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MNIST  –  Data loading & preprocessing")
    print("=" * 60)

    print("\n[1/3] Downloading sprite sheets …")
    X_train, X_test, y_train, y_test = load_mnist()

    print("\n[2/3] Dataset summary")
    print(f"  Total images     : {len(y_train) + len(y_test):>6}")
    print(f"  Training set     : {len(y_train):>6}  images  –  X_train shape: {X_train.shape}")
    print(f"  Test set         : {len(y_test):>6}  images  –  X_test  shape: {X_test.shape}")
    print(f"  Feature dimension: {X_train.shape[1]:>6}  (= 28 × 28 pixels, flattened)")
    print(f"  Pixel range      : [{X_train.min():.2f}, {X_train.max():.2f}]  (normalised)")
    print(f"  Classes          : {sorted(set(y_train.tolist()))}")

    print("\n[3/3] Class distribution in training set")
    for digit in range(10):
        count = int((y_train == digit).sum())
        print(f"  Digit {digit}: {count} images")

    print("\nAll done.  X_train, X_test, y_train, y_test are ready.")
