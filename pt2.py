import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def load_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    X = batch[b'data']
    y = batch[b'labels']
    return X, y

data_dir = "pt2-batches-py"

X_train_list, y_train_list = [], []
for i in range(1, 6):
    X, y = load_batch(os.path.join(data_dir, f"data_batch_{i}"))
    X_train_list.append(X)
    y_train_list.extend(y)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.array(y_train_list)

X_test, y_test_list = load_batch(os.path.join(data_dir, "test_batch"))
y_test = np.array(y_test_list)

print(X_train.shape)                                                        # (50000, 3072)
print(X_test.shape)                                                         # (10000, 3072)

def reshape_cifar(X):
    X = X.reshape(-1, 3, 32, 32)
    X = X.transpose(0, 2, 3, 1)
    return X / 255.0

X_train_rgb = reshape_cifar(X_train)
X_test_rgb  = reshape_cifar(X_test)

def to_grayscale(X_rgb):
    return 0.299 * X_rgb[:,:,:,0] + 0.587 * X_rgb[:,:,:,1] + 0.114 * X_rgb[:,:,:,2]

X_train_gray = to_grayscale(X_train_rgb)
X_test_gray  = to_grayscale(X_test_rgb)

print(X_train_gray.shape)                                                   # (50000, 32, 32)
print(X_test_gray.shape)                                                    # (10000, 32, 32)





fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("RGB vs Grayscale", fontsize=14)

for i in range(5):
    axes[0, i].imshow(X_train_rgb[i])
    axes[0, i].set_title(f"RGB #{i}")
    axes[0, i].axis('off')

    axes[1, i].imshow(X_train_gray[i], cmap='gray')
    axes[1, i].set_title(f"Gray #{i}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("verification_grayscale.png", dpi=150)
plt.show()