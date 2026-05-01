import numpy as np


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

    def gradient_descent(self):
        print("coming soon")

"""
# Mon exemple de test
model = LinearXSoftmax(input_size=784, num_classes=10)
#X_test = np.random.rand(1, 784)  # on simule une image
predictions = model.forward(X_test[0])  # on test

print("Proba pour chaque chiffre (0 à 9) :")
print(predictions)
print(f"Somme des proba : {np.sum(predictions)}")  # Doit être environ égal à 1.0 si tt est ok
"""