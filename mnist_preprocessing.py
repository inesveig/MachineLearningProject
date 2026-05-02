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

import matplotlib.pyplot as plt
from Part1 import LinearXSoftmax, apply_pca, evaluate_model

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
    cols = 40 # actual_w // TILE_SIZE  # integer division — ignores any remainder
    rows = 25 # actual_h // TILE_SIZE

    patches = []

    # through manual checking we note that there are EXACTLY 358 pixels between every 10 images horizontally
    # matter of fact there's also 309 pixels between every 10 images vertically
    # and since this isn't a multiple of 10 we need to work around this
    ver_total_offset = 197 # hard coded offset values because the number grid on the Github images don't start at the
    hor_total_offset = 304 # very top-left corner so (0,0) is an incorrect starting point and messes this up

    for row in range(rows):
        sec = 0 # also known as the Seven/Eight (horizontal) Counter which resets at the end of each line
        for col in range(cols):
            x0 = hor_total_offset + sec + col * TILE_SIZE
            x1 = hor_total_offset + sec + (col + 1) * TILE_SIZE

            if col % 5 == 2: # there's a 7/8 pixel gap between images horizontally
                sec += 7
            else:
                sec += 8

            y0 = ver_total_offset + 3 * row + row * TILE_SIZE # + 3*row because there's a 3px gap vertically
            y1 = ver_total_offset + 3 * row + (row + 1) * TILE_SIZE

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

    plt.figure(figsize=(10, 3))
    for i in range(4):
        X = X_train[i].reshape(28, 28)
        y = y_train[i]
        plt.subplot(1, 4, i + 1)
        plt.imshow(X, cmap="gray")
        plt.title(f"Label: {y}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

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
    # Mon exemple de test
    model = LinearXSoftmax(input_size=784, num_classes=10)
    # X_test = np.random.rand(1, 784)  # on simule une image

    X = X_train  #shape (8000, 784)
    y = y_train  #shape (8000,)
    predictions = model.forward(X_train) #shape (8000, 10) -> proba de chaque nombre pour chaque image
    pred_vec = model.prediction(X) #shape(8000,)
    error = model.compute_loss(y, predictions)


    plt.figure(figsize=(10, 3))
    for i in range(4):
        print("\nProba pour chaque chiffre (0 à 9) :")
        print(predictions[i])
        #print(f"Somme des proba : {np.sum(predictions)}")  # Doit être environ égal à 1.0 si tt est ok
        X_disp = X[i].reshape(28, 28)
        y_disp = y[i]
        plt.subplot(1, 4, i + 1)
        plt.imshow(X_disp, cmap="gray")
        plt.title(f"Prediction:{pred_vec[i]} Label: {y_disp}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()





    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test, n_components=50)

    model = LinearXSoftmax(input_size=X_train_pca.shape[1], num_classes=10)
    model.train(X_train_pca, y_train, learning_rate=0.1, epochs=100)

    train_accuracy, train_error_rate, _ = evaluate_model(model, X_train_pca, y_train)
    test_accuracy, test_error_rate, _ = evaluate_model(model, X_test_pca, y_test)

    print(f"Training error rate: {train_error_rate:.2f}%")
    print(f"Test error rate: {test_error_rate:.2f}%")






