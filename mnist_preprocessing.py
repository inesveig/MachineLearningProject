"""
MNIST Data Loading & Preprocessing  –  Section 1.1
===================================================
Source: https://github.com/mbornet-hl/MNIST/tree/master/IMAGES/GROUPS
"""

import io
import urllib.request

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from Part1 import LinearXSoftmax, MultiLayerPerceptron, apply_pca, evaluate_model
from part1hiddenlayers import MLP
from sklearn.decomposition import PCA

# ── Constants ────────────────────────────────────────────────────────────────

TILE_SIZE = 28
GRID_COLS = 25
GRID_ROWS = 40
N_PER_CLASS = GRID_COLS * GRID_ROWS

# The raw-file URL template.  GitHub serves raw PNGs at this base path.
RAW_BASE = ("https://raw.githubusercontent.com/mbornet-hl/MNIST/master/IMAGES/GROUPS/")

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

TEST_SIZE = 0.20    # 20 % of total data reserved for testing
RANDOM_SEED = 42      # fixed seed → reproducible split across runs


def fetch_image(url: str) -> Image.Image:
    with urllib.request.urlopen(url) as response:
        raw_bytes = response.read()
    return Image.open(io.BytesIO(raw_bytes))


def slice_sprite_sheet(sheet: Image.Image, digit: int) -> tuple[np.ndarray, np.ndarray]:

    sheet_gray = sheet.convert("L")
    sheet_arr = np.array(sheet_gray, dtype=np.float32)

    cols = 40 # actual_w // TILE_SIZE
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


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    all_X, all_y = [], []

    for digit, filename in enumerate(FILENAMES):
        url = RAW_BASE + filename
        print(f"  Downloading digit {digit}  ←  {filename} …", end=" ", flush=True)

        sheet = fetch_image(url)
        X_class, y_class = slice_sprite_sheet(sheet, digit)
        all_X.append(X_class)
        all_y.append(y_class)
        print(f"done  ({X_class.shape[0]} images)")

    # we stack all classes into one dataset
    X = np.vstack(all_X)   # (10 000, 784)
    y = np.concatenate(all_y)       # (10 000,)

    # we shuffle and split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,          # preserve class balance in both sets
    )

    return X_train, X_test, y_train, y_test


# MAIN
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
    model2 = MultiLayerPerceptron(input_size=784, num_classes=10)
    # X_test = np.random.rand(1, 784)  # on simule une image

    X = X_train  #shape (8000, 784)
    y = y_train  #shape (8000,)
    predictions = model2.forward(X_train) #shape (8000, 10) -> proba de chaque nombre pour chaque image
    pred_vec = model2.prediction(X) #shape(8000,)
    error = model2.compute_loss(y, predictions)


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

    print("\nPCA output:")
    print(f"X_train_pca shape: {X_train_pca.shape}")
    print(f"X_test_pca shape: {X_test_pca.shape}")

    print("\nPCA du train:")
    print(X_train_pca[0])

    print("\nPCA du test:")
    print(X_test_pca[0])

    pca_vis = PCA(n_components=2)
    X_train_pca_2d = pca_vis.fit_transform(X_train)
    X_test_pca_2d = pca_vis.transform(X_test)


    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_train_pca_2d[:, 0],
        X_train_pca_2d[:, 1],
        c=y_train,
        cmap="tab10",
        s=10
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA du training set")
    plt.colorbar(scatter, label="Classe")
    plt.show()


    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_test_pca_2d[:, 0],
        X_test_pca_2d[:, 1],
        c=y_test,
        cmap="tab10",
        s=10
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA du test set")
    plt.colorbar(scatter, label="Classe")
    plt.show()


    model2 = MultiLayerPerceptron(input_size=X_train_pca.shape[1], num_classes=10)
    model2.train(X_train_pca, y_train, learning_rate=0.1, epochs=100)

    train_accuracy, train_error_rate, _ = evaluate_model(model2, X_train_pca, y_train)
    test_accuracy, test_error_rate, _ = evaluate_model(model2, X_test_pca, y_test)

    print(f"\nTraining error rate: {train_error_rate:.2f}%")
    print(f"Test error rate: {test_error_rate:.2f}%")






