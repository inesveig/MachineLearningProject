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

