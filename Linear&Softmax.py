import numpy as np


class LinearXSoftmax:
    def __init__(self, input_size, num_classes): #voir avec le travail des autres
        self.W = np.random.randn(input_size, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))

    def softmax(self, z):
        # on transforme les scores en proba
        # on soustrait le max de chaque ligne
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # 1. les scores z = XW + b
        self.z = np.dot(X, self.W) + self.b

        # 2. Softmax
        self.probs = self.softmax(self.z)
        return self.probs

    def compute_loss(self, y_true, probs):  # pour cross entropy
        m = y_true.shape[0]  # nb d'image
        loss = -1 / m * np.sum(y_true * np.log(probs + 1e-8))  # ajouter un 1e-8 pour éviter une erreur avec un Log
        return loss


# Mon exemple de test
model = LinearXSoftmax(input_size=784, num_classes=10)
X_test = np.random.rand(1, 784)  # on simule une image
predictions = model.forward(X_test)  # on test

print("Proba pour chaque chiffre (0 à 9) :")
print(predictions)
print(f"Somme des proba : {np.sum(predictions)}")  # Doit être environ égal à 1.0 si tt est ok
