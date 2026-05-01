import numpy as np                                  #To manipulate multi-dimensionality tab

import pickle
import os                                           #To acess the files
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pt23 import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    X = batch[b'data']
    y = batch[b'labels']
    return X, y

data_dir = "pt2-batches-py"

with open("pt2-batches-py/batches.meta", "rb") as fi:
    dict = pickle.load(fi, encoding='bytes')

print(dict)
label_names = dict[b'label_names'] # array containing all the labels corresponding to predictions
label_names_clean = []
for label in label_names:
    label = str(label)[2:-1]
    label_names_clean.append(label)


X_train_list, y_train_list = [], []
for i in range(1, 6):
    X, y = load_batch(os.path.join(data_dir, f"data_batch_{i}"))
    X_train_list.append(X)
    y_train_list.extend(y)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.array(y_train_list) # (50000, )

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


#{----------------RGB image---------------}
# Apply filters on RGB images directly (part 2.3.3)

"""
K_R = np.array([[-6, 1, 0], [2, 7, -4], [0, 0, 3]])
K_G = np.array([[3, 0, -4], [1, 1, 1], [-2, 6, 9]])
K_B = np.array([[0, 2, 10], [4, 8, -6], [-5, -3, -7]])
bias = 0
"""
num_filters = 64

K_R = np.random.randn(num_filters, 3, 3)
K_G = np.random.randn(num_filters, 3, 3)
K_B = np.random.randn(num_filters, 3, 3)

bias = np.random.randn(num_filters)

img = X_train_rgb[1]

padded_img = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
img_h, img_w, channels = img.shape

output = np.zeros((num_filters, img_h, img_w), dtype=np.float32)

# Slide the filters over the image
for i in range(num_filters):
    for y in range(img_h):
        for x in range(img_w):
            # Extract the 3x3 patch for each color channel
            patch_R = padded_img[y:y + 3, x:x + 3, 0]  # Red channel
            patch_G = padded_img[y:y + 3, x:x + 3, 1]  # Green channel
            patch_B = padded_img[y:y + 3, x:x + 3, 2]  # Blue channel

            # Element-wise multiplication and sum for EACH channel
            sum_R = np.sum(patch_R * K_R[i])
            sum_G = np.sum(patch_G * K_G[i])
            sum_B = np.sum(patch_B * K_B[i])

            # Add them all together with the bias (Mathematical formula page 13)
            pixel_value = sum_R + sum_G + sum_B + bias[i]
            output[i, y, x] = pixel_value


fig, axes = plt.subplots(1, 3, figsize=(10, 5))

# Original color image (Need to cast back to integer 0-255 for matplotlib to display RGB correctly)
axes[0].imshow(img)
axes[0].set_title('Original Color Image (3D Volume)')
axes[0].axis('off')

# The resulting 2D feature map
axes[1].imshow(output[0])
axes[1].set_title('Output First Feature Map (2D Grid)')
axes[1].axis('off')

axes[2].imshow(output[63])
axes[2].set_title('Output Last Feature Map (2D Grid)')
axes[2].axis('off')

plt.tight_layout()
plt.show()


#{----------------Max Pooling---------------}

pool_size = 2
stride = 2

num_filters, H, W = output.shape

pooled_H = H // 2
pooled_W = W // 2

pooled = np.zeros((num_filters, pooled_H, pooled_W))

for f in range(num_filters):
    for y in range(pooled_H):
        for x in range(pooled_W):
            # zone 2x2
            patch = output[
                    f,
                    y * stride:y * stride + pool_size,
                    x * stride:x * stride + pool_size
                    ]

            pooled[f, y, x] = np.max(patch)


fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(output[0])
axes[0].set_title("Before pooling (32x32)")

axes[1].imshow(pooled[0])
axes[1].set_title("After pooling (16x16)")

for ax in axes:
    ax.axis('off')

plt.show()

#{-------------------Model-----------------}
# X_train: (50000, 3072) OR (50000, 32, 32, 3)
# Y_train: (50000,)

X = torch.tensor(X_train, dtype=torch.float32)
y = torch.tensor(y_train, dtype=torch.long)

X = X.view(-1, 3, 32, 32)

X = torch.mean(X, dim=1, keepdim=True)  # (50000, 1, 32, 32)

print("Shape finale X:", X.shape)  # doit être (50000, 1, 32, 32)



# DATA LOADER

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # (32, 32, 32)
        self.pool = nn.MaxPool2d(2, 2)                            # /2

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (64, 16, 16)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (64, 8, 8)

        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = CNN()



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



epochs = 10
correct = 0
total = 0

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f}")

accuracy = 100 * correct / total
print(f"Training Accuracy: {accuracy:.2f}%")

model.eval()

X_test = torch.tensor(X_test, dtype=torch.float32)
X_test = X_test.view(-1, 3, 32, 32)
X_test = torch.mean(X_test, dim=1, keepdim=True)
y_test = torch.tensor(y_test, dtype=torch.long)


with torch.no_grad():
    outputs = model(X_test)
    _, predictions = torch.max(outputs, 1)

correct = (predictions == y_test).sum().item()
accuracy = 100 * correct / len(y_test)


print(f"Test Accuracy: {accuracy:.2f}%")

i = 6

img = X_test[i][0].numpy()

plt.imshow(img, cmap="gray")
pred_num = predictions[i].item()
pred_name = label_names_clean[pred_num]
true_num = y_test[i].item()
true_name = label_names_clean[true_num]

plt.title(f"Pred: {pred_num, pred_name} | True label: {true_num, true_name}")
plt.axis("off")
plt.show()
