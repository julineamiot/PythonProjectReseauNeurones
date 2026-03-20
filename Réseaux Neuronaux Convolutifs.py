"""appliquer 4 filtres différents sur l'image
on va donc obtenir 4 images différentes selon le filtre appliqué
normaliser l'image
multiplier la valeur du pixel par la valeur du filtre
centrer le filtre et faire addition des multiplications et obtenir un seul chiffre
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

    def padding(self, image, epaisseur):
        # on agrandit la matrice
        return None

    def convolution(self, image, filtre):
        # image : matrice 28x28
        # filtre : matrice 3x3
        # à chaque position on multiplie les pixels de l'image par les poids du filtre et on fait la somme
        # sortie : matrice de taille r
        return None

    def max_pooling(self, Image_2):
        pass #renvoie matrice (sortie_pooling) avec comme coeff la valeur max de cahque Image_2

    def dense(self,sortie_pooling):
        pass #donne un vecteur (vect_dense)

    """"def forwardPropag(self, imageMatrice):
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

        return activation, zs"""

    """def backPropag(self, imageMatrice, label):
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
            self.biais[l] -= self.learning_rate * deltas[l]"""

class MnistDataloader(object):

    def __init__(self): # training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        # à changer en fonction de vos chemins d'accès sur vos ordinateurs
        #input_path = "/Users/julineamiot/Documents/PycharmProjects/PythonProjectReseauNeurones"
        input_path = r"C:\Users\Utilisateur\OneDrive\Documents\Cours\TSE\L3\Programmation, magistère\Projet"
        training_images_filepath = input_path + "/train-images.idx3-ubyte"
        training_labels_filepath = input_path + "/train-labels.idx1-ubyte"
        test_images_filepath = input_path + "/t10k-images.idx3-ubyte"
        test_labels_filepath = input_path + "/t10k-labels.idx1-ubyte"


        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(rows, cols)
            images.append(img)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


# vérifie la lecture du Dataset via la classe MnistDataloader
#Affichage images
# Fonction utilitaire pour afficher une liste d’images avec leurs titres correspondants

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15)
        index += 1
# charger MINST dataset

if __name__ == "__main__":
    # 1. Charger les données (assure-toi que tes fichiers sont au bon endroit)
    mnist_dataloader = MnistDataloader()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # 2. Initialiser le réseau
    reseau = ReseauNeurones(nbNeuronesCouche)
    reseau.initialiserPoids()

    # 3. Entraînement (sur un petit échantillon pour tester)
    #print("Début de l'entraînement...")
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

    print("Précision globale : " + str((correct / nb_tests) * 100) + "%")