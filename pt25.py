import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle



def load_batch(filepath):
    with open(filepath, "rb") as f:
        batch = pickle.load(f, encoding = "bytes")
    return batch[b"data"], batch[b"labels"]



def load_cifar10(data_dir):
    X_train_list = []
    y_train_list = []
    for i in range(1,6):
        X, y = load_batch(os.path.join(data_dir, "data_batch_" + str(i)))
        X_train_list.append(X)
        y_train_list.extend(y)
    
    X_train = np.concatenate(X_train_list, axis = 0)
    y_train = np.array(y_train_list)

    X_test, y_test_list = load_batch(os.path.join(data_dir, "test_batch"))
    y_test = np.array(y_test_list)

    return X_train, y_train, X_test, y_test


def reshape_cifar(X):
    X = X.reshape(-1, 3, 32, 32)            #to get (N, 3072) -> (N, 3, 32, 32)
    return X/255



data_dir = "pt2-batches-py"
X_train_raw, y_train, X_test_raw, y_test = load_cifar10(data_dir)
print("a")
X_train = reshape_cifar(X_train_raw).astype(np.float32)             # (50000, 3, 32, 32)
X_test = reshape_cifar(X_test_raw).astype(np.float32)               # (10000, 3, 32, 32)
print("b")            



