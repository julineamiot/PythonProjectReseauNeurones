"""appliquer 4 filtres différents sur l'image
on va donc obtenir 4 images différentes selon le filtre appliqué
normaliser l'image
multiplier la valeur du pixel par la valeur du filtre
centrer le filtre et faire addition des multiplications avec biais et obtenir un seul chiffre
faire du pading à 0
le but est d'obtenir des nouveaux paterne sur l'image
stride :  de combien on décale la matrice de filtre
le réseau de neuronne va apprendre les chiffres qui sont dans le filtre
permet de compresser le nb dinformations à connaitre
après la convolution il y a une fonction d'activation de type Relu (appliqué sur chaque éléments de sortie de la convolution)
pooling : réduire l'information et résumer l'information (Average Pooling = moyenne, + Max Pooling = grande valeur (c'est le + utilisé))
Flatten : on doit mettre en ligne toutes les infos pour que la phase dense puisse les utiliser
dense : mettre toutes les info en un veteur (si beaucoup d'image on peut faire 2 fois cette étape) le nb de dense correspond au nb de neurones que l'on doit avoir en sortie
pour éviter le sur apprentissage (= Overfitting), on peut tourner, flouter l'image
carte de saillance : dire que l'on a beaucoup utiliser pour détecter l'objet (repose sur le calcul du gradient)

1ere couche 64 neurones puis 128 pour les suivantes

(convolution => activation (Relu) => pooling) * (x fois) => dense => prediction"""

import struct
from array import array
import random
import matplotlib.pyplot as plt
import numpy as np

# Pour reconnaître tous les chiffres, on met 10 neurones en sortie
nbNeuronesCouche = [784, 64, 10]
learning_rate = 0.01


class ReseauNeurones:
    def __init__(self, nbNeuronesCouche):
        self.tailles = nbNeuronesCouche
        self.nbCouches = len(nbNeuronesCouche)
        self.poids = []
        self.biais = []
        self.learning_rate = learning_rate

    def ReLuActivation(self, x):
        return np.maximum(0, x)

    def ReLuPrime(self, x):
        return np.where(x < 0, 0, 1)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum()

    def initialiserPoids(self):
        for i in range(self.nbCouches - 1):
            p = np.random.randn(self.tailles[i], self.tailles[i + 1]) * np.sqrt(2 / self.tailles[i])
            b = np.zeros(self.tailles[i + 1])
            self.poids.append(p)
            self.biais.append(b)

    def forwardPropag(self, imageMatrice):
        # On aplatit l'image 28x28 en vecteur de 784
        pix = imageMatrice.flatten() / 255.0
        activation = [pix]
        zs = []

        for i in range(len(self.poids)):
            z = np.dot(activation[-1], self.poids[i]) + self.biais[i]
            zs.append(z)

            if i == len(self.poids) - 1:
                # Dernière couche Softmax
                a = self.softmax(z)
            else:
                # Couches cachées ReLU
                a = self.ReLuActivation(z)
            activation.append(a)

        return activation, zs

    def backPropag(self, imageMatrice, label):
        activation, zs = self.forwardPropag(imageMatrice)

        cible = np.zeros(10) # vecteur cible
        cible[label] = 1

        deltas = [None] * len(self.poids)

        deltas[-1] = activation[-1] - cible

        # Rétropropagation de l'erreur
        for l in reversed(range(len(self.poids) - 1)):
            deltas[l] = np.dot(deltas[l + 1], self.poids[l + 1].T) * self.ReLuPrime(zs[l])

        # Mise à jour des poids et biais
        for l in range(len(self.poids)):
            # On utilise np.outer pour multiplier le vecteur d'entrée par le vecteur d'erreur
            self.poids[l] -= self.learning_rate * np.outer(activation[l], deltas[l])
            self.biais[l] -= self.learning_rate * deltas[l]


if __name__ == "__main__":
    # 1. Charger les données (assure-toi que tes fichiers sont au bon endroit)
    mnist_dataloader = MnistDataloader()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # 2. Initialiser le réseau
    reseau = ReseauNeurones(nbNeuronesCouche)
    reseau.initialiserPoids()

    # 3. Entraînement (sur un petit échantillon pour tester)
    print("Début de l'entraînement...")
    for i in range(5):
        for image, label in zip(x_train[:2000], y_train[:2000]):
            reseau.backPropag(image, label)

    # 4. Test final
    correct = 0
    nb_tests = 1000

    for image, label in zip(x_test[:nb_tests], y_test[:nb_tests]):
        activations, _ = reseau.forwardPropag(image)
        # La prédiction est l'indice du neurone qui a le plus gros score
        prediction = np.argmax(activations[-1])

        if prediction == label:
            correct += 1

    print("Précision globale :" + str((correct / nb_tests) * 100) + "%")