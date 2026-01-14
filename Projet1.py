import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image

nbNeuronesCouche = [784, 16, 1] #3 couches, 1ere couche 784 neurones, 2e couche 16 neurones, 3e couche 1 neurone car doit dire si c'est un x ou  pas
X = 3 # chiffre que le programme doit apprendre/reconnaitre

class ReseauNeurones:
    def __init__(self, nbNeuronesCouche):
        self.tailles = nbNeuronesCouche
        self.nbCouches = len(nbNeuronesCouche)
        self.poids = []
        self.biais = []
        self.learning_rate = 0.005

    def ReLuActivation(self, x):
        return np.where(x<0, 0, x)

    def ReLuPrime(self, x):
        return np.where(x<0, 0, 1)

    def sigmoideActivation(self, x): # on utilise sigmoide pour la couche de sortie
        return 1 / (1 + np.exp(-x))

    def sigmoidePrime(self, x):
        s = self.sigmoideActivation(x)
        return s * (1 - s)

    """def ouvrirImage(self):
        #Cette fonction ouvre une image et la convertit en matrice numpy,  avec des niveaux de gris entre 0 et 255.
        image = Image.open("nom de l'image")
        imageGris = image.convert("L")
        imageMatrice = np.asarray(imageGris)
        return imageMatrice"""

    def initialiserPoids(self):
        for i in range(self.nbCouches - 1): #-1 car les poids relient les couches entre elles
            poids = np.random.randn(self.tailles[i], self.tailles[i + 1]) * np.sqrt(2 / self.tailles[i])
            biais = np.zeros(self.tailles[i + 1])
            self.poids.append(poids)
            self.biais.append(biais)

    def forwardPropag(self, imageMatrice):
        pix = imageMatrice.reshape(-1) / 255 # on a des images 28x28 donc on appalatit l'image en vecteur 784 normalisés (ie avec des valeurs proches de 0)
        activation = [pix] # valeurs après l'activation
        zs = [] # valeurs avant l'activation (pour la backward)

        for i in range(len(self.poids)):
            z = np.dot(pix, self.poids[i]) + self.biais[i]
            zs.append(z)

            if i == len(self.poids) - 1:
                pix = self.sigmoideActivation(z)
            else:
                pix = self.ReLuActivation(z)

            activation.append(pix)

        return activation, zs

    def backPropag(self, imageMatrice, label):
        activation, zs = self.forwardPropag(imageMatrice)

        # cible pour un seul neurone de sortie
        cible = 1 if label == X else 0  # 1 si c'est le chiffre X (cf. début programme), sinon 0

        deltas = [None] * len(self.poids)  # liste des deltas (une par couche de poids)

        deltas[-1] = activation[-1] - cible # delta (=erreur) de la couche de sortie

        # on fait la backward pour les couches cachées
        for l in reversed(range(len(self.poids) - 1)):
            deltas[l] = np.dot(deltas[l + 1], self.poids[l + 1].T) * self.ReLuPrime(zs[l])

        # on met à jour les poids et les biais
        for l in range(len(self.poids)):
            a = activation[l].reshape(-1, 1)  # activation de la couche l
            d = deltas[l].reshape(1, -1)  # delta de la couche l+1
            self.poids[l] = self.poids[l] - self.learning_rate * np.dot(a, d)
            self.biais[l] = self.biais[l] - self.learning_rate * deltas[l] #on met à jour le biais

class MnistDataloader(object):

    def __init__(self): # training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        # à changer en fonction de vos chemins d'accès sur vos ordinateurs
        input_path = "/Users/julineamiot/PycharmProjects/PythonProjectReseauNeurones"
        # input_path = r"C:\Users\Utilisateur\OneDrive\Documents\Cours\TSE\L3\Programmation, magistère\Projet"
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


if __name__=="__main__":
    mnist_dataloader = MnistDataloader()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Afficher quelques images aléatoires
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

    show_images(images_2_show, titles_2_show)

    # on initialise le réseau
    reseau = ReseauNeurones(nbNeuronesCouche)
    reseau.initialiserPoids()

    # entraînement
    print("Entraînement du réseau :")
    for epoch in range(30):
        for image, label in zip(x_train[:100], y_train[:100]):
            reseau.backPropag(image, label)

    # test
    print("\nTest du réseau")
    correct = 0

    for image, label in zip(x_test, y_test):
        resultat = reseau.forwardPropag(image)
        activations = resultat[0]
        sortie = activations[-1][0]

        prediction = 1 if sortie > 0.5 else 0
        cible = 1 if label == X else 0

        if prediction == cible:
            correct = correct + 1
    tauxReussite = correct / len(x_test) * 100
    print("Taux de réussite pour détecter le chiffre " + str(X) + " : " + str(tauxReussite) +"%")