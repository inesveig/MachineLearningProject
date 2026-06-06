import numpy as np
from sklearn.decomposition import PCA

class LinearXSoftmax:
    def __init__(self, input_size, num_classes): #voir avec le travail des autres
        self.W = np.random.randn(input_size, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))

    def softmax(self, z):
        # on transforme les scores en proba
        # on soustrait le max de chaque ligne
        exp_z = np.exp(z) # - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # 1. les scores z = XW + b
        self.z = np.dot(X, self.W) + self.b

        # 2. Softmax
        self.probs = self.softmax(self.z)
        return self.probs

    def prediction(self, X): # find index du maximum (= chiffre prédit)
        return np.argmax(self.forward(X), axis=1)

    def compute_loss(self, true_labels, predicted_labels):  # pour cross entropy
        # true_labels = vector of n rows
        n = true_labels.shape[0]
        # récupérer la proba de la bonne classe pour chaque exemple
        correct_logprobs = np.log(predicted_labels[np.arange(n), true_labels])
        loss = (-1/n) * np.sum(correct_logprobs)
        return loss

    """
    # Mon exemple de test
    model = LinearXSoftmax(input_size=784, num_classes=10)
    #X_test = np.random.rand(1, 784)  # on simule une image
    predictions = model.forward(X_test[0])  # on test
    
    print("Proba pour chaque chiffre (0 à 9) :")
    print(predictions)
    print(f"Somme des proba : {np.sum(predictions)}")  # Doit être environ égal à 1.0 si tt est ok
    """


    
    def one_hot(self, y, num_classes): #transformation en matrice de 0 et de 1
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot


    def gradient_descent(self, X, y):
        n = X.shape[0]
        Y = self.one_hot(y, self.b.shape[1])

        #Les partial derivative
        dZ = self.probs - Y
        dW = (1 / n) * np.dot(X.T, dZ)
        db = (1 / n) * np.sum(dZ, axis=0, keepdims=True)

        return dW, db

    def train(self, X, y, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            probs = self.forward(X)
            loss = self.compute_loss(y, probs)
            dW, db = self.gradient_descent(X, y)
            
            self.W -= learning_rate * dW
            self.b -= learning_rate * db

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")


class MultiLayerPerceptron:
    """
    In terms of architecture we have 784  →  256  →  128  →  10 (neurons)
    For activation functions we have regular ReLU for the 2 hidden layers and Softmax for the output(s)
    As for loss, Cross-entropy H(p, q)
    Finally for learning we have Mini-batch gradient descent + full backpropagation (weights and biases)

    As for notation:
        a = weight matrix for each layer (rows are the neurons on layer h-1 and columns are the neurons on layer h)
        b = bias vector for each layer
        o = score or logit matrix (= a @ input + b) with @ being the product
        z = neuron output matrix after activation, the same size as o
    """

    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=10):
        self.a1 = np.random.randn(input_size, hidden1) * np.sqrt(2 / input_size) # (784, 256) initialized randomly and scaled down, "input size" rows and "number of neurons in layer 1" columns
        self.b1 = np.zeros((1, hidden1)) # (1, 256) one bias per neuron in layer 1

        self.a2 = np.random.randn(hidden1, hidden2) * np.sqrt(2 / hidden1) # same as above except it's (256, 128)
        self.b2 = np.zeros((1, hidden2)) # (1, 128) one bias per neuron in layer 2

        self.a3 = np.random.randn(hidden2, num_classes) * np.sqrt(2 / hidden2) # (128, 10) "n° of neurons in layer 2" rows and "n° of output class" columns
        self.b3 = np.zeros((1, num_classes)) # (1,  10) one bias per output class

    def relu(self, o):
        return np.maximum(0, o)

    def relu_prime(self, o):
        return (o > 0).astype(float) # derivative of ReLU is 1 above 0 and 0 under 0

    def softmax(self, o):
        shifted = o - np.max(o, axis=1, keepdims=True) # shifted and this doesn't change the softmax output because
        exp_o = np.exp(shifted) # e^(x-c) / sum(e^(x-c)) = e^x / sum(e^x)
        return exp_o / np.sum(exp_o, axis=1, keepdims=True)

    def forward(self, X): # forward pass to get the output.s from X

        # Layer 1, o and z are shape (n, 256)
        self.o1 = np.dot(X, self.a1) + self.b1
        self.z1 = self.relu(self.o1)

        # Layer 2, shape (n, 128)
        self.o2 = np.dot(self.z1, self.a2) + self.b2
        self.z2 = self.relu(self.o2)

        # Output layer, shape (n,  10)
        self.o3 = np.dot(self.z2, self.a3) + self.b3
        self.z3 = self.softmax(self.o3)

        return self.z3

    def prediction(self, X):
        return np.argmax(self.forward(X), axis=1) # returns the highest score

    def one_hot(self, y, num_classes):
        """
        Example for num_classes=10 and y=[2, 0, 7]:
            row 0 → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            row 1 → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            row 2 → [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        """
        onehot = np.zeros((y.shape[0], num_classes))
        onehot[np.arange(y.shape[0]), y] = 1 # for each row i it sets column y[i] to 1, we need this for cross entropy
        return onehot

    def compute_loss(self, true_labels, predicted_probs):
        # Cross-entropy loss which is the same as in LinearXSoftmax.
        n = true_labels.shape[0]
        probs_clipped = np.clip(predicted_probs, 1e-12, 1.0)
        correct_logprobs = np.log(probs_clipped[np.arange(n), true_labels])
        return (-1 / n) * np.sum(correct_logprobs)

    def backpropagation(self, X, y): # backpropagation
        """
        Compute gradients for every weight and bias via the chain rule.

        Y = one_hot(y): shape (n, 10). We need the full matrix form of
            the true labels to compute the gradient do3 = z3 - Y which
            has shape (n, 10)

        Notation for gradients:
            do3  – gradient of the loss with respect to the scores o3
            da3  – gradient of the loss with respect to the weight matrix a3
            db3  – gradient of the loss with respect to the bias vector b3
            dz2  – gradient of the loss with respect to the layer-2 outputs z2
            same for layer 1
        """
        n = X.shape[0]
        Y = self.one_hot(y, self.b3.shape[1])   # (n, 10)

        # Output layer, the gradient of (softmax + cross-entropy) simplifies to z3 - Y
        do3 = self.z3 - Y                                     # (n, 10), like o3
        da3 = (1 / n) * np.dot(self.z2.T, do3)                # (128, 10), like a3
        db3 = (1 / n) * np.sum(do3, axis=0, keepdims=True)    # (1, 10), like b3

        # Hidden layer h = 2, we backpropagate from the output layer h = 3 with a special formula
        dz2 = np.dot(do3, self.a3.T)                          # (n, 128)
        do2 = dz2 * self.relu_prime(self.o2)                  # (n, 128)
        da2 = (1 / n) * np.dot(self.z1.T, do2)                # (256, 128)
        db2 = (1 / n) * np.sum(do2, axis=0, keepdims=True)    # (1, 128)

        # Hidden layer h = 1
        dz1 = np.dot(do2, self.a2.T)                          # (n, 256)
        do1 = dz1 * self.relu_prime(self.o1)                  # (n, 256)
        da1 = (1 / n) * np.dot(X.T, do1)                      # (784, 256)
        db1 = (1 / n) * np.sum(do1, axis=0, keepdims=True)    # (1, 256)

        return da1, db1, da2, db2, da3, db3

    def train(self, X, y, learning_rate=0.01, epochs=100, batch_size=16):
        # training with mini-batch descent
        """
        Why mini-batches instead of the full dataset?
            With ~235k parameters, computing gradients over all 8000 samples
            every step is slow
        """
        n = X.shape[0]

        for epoch in range(epochs):
            # Shuffle training data each epoch so that the model doesn't learn the order of the data instead of trend
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            n_batches = 0

            for start in range(0, n, batch_size): # we go batch by batch
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                # Forward
                probs = self.forward(X_batch)
                epoch_loss += self.compute_loss(y_batch, probs)
                n_batches += 1

                # Backward
                da1, db1, da2, db2, da3, db3 = self.backpropagation(X_batch, y_batch)
                self.a1 -= learning_rate * da1
                self.b1 -= learning_rate * db1
                self.a2 -= learning_rate * da2
                self.b2 -= learning_rate * db2
                self.a3 -= learning_rate * da3
                self.b3 -= learning_rate * db3

            if epoch % 10 == 0: # compute average loss every 10 epochs
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch:>4},  Loss: {avg_loss:.4f}")


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


def compute_error_rate(y_true, y_pred):
    return 100 - compute_accuracy(y_true, y_pred)


def evaluate_model(model, X, y):
    y_pred = model.prediction(X)
    accuracy = compute_accuracy(y, y_pred)
    error_rate = compute_error_rate(y, y_pred)
    return accuracy, error_rate, y_pred


def apply_pca(X_train, X_test, n_components=50):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

