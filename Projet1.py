import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

""" Instructions : le programme doit prendre en entrée une matrice np de taille paramétrable. Le réseau doit avoir une couche d'entrée de la taille de 
l'image et à la fin on a une sortie. Pour chaque couche, faire produit matriciel entre les indices de la matrice avec poids. 
Si valeur du neurone inféireur à 0, on renvoie -1, sinon on renvoie 1"""

nbNeuronesCouche = [784, 64, 32, 1] #4 couches, 1ere couche 784 neurones, 2e couche 64 neurones, 3e couche 32 neurones, 4e couche 1 neurone car doit dire si c'est un x ou  pas

class ReseauNeurones:
    def __init__(self, nbNeuronesCouche):
        self.tailles = nbNeuronesCouche
        self.nbCouches = len(nbNeuronesCouche)
        self.poids = []
        self.learning_rate = 0.01

    def fonctionActivation(self, x):
        return np.where(x<0, 0, x)

    def deriveeActivation(self, x):
        return np.where(x<0, 0, 1)

    def ouvrirImage(self):
        #Cette fonction ouvre une image et la convertit en matrice numpy,  avec des niveaux de gris entre 0 et 255.
        image = Image.open("nom de l'image")
        imageGris = image.convert("L")
        imageMatrice = np.asarray(imageGris)
        return imageMatrice

    def initialiserPoids(self):
        for i in range(self.nbCouches - 1): #-1 car les poids relient les couches entre elles
            poids = np.random.uniform(-1, 1,(self.tailles[i], self.tailles[i + 1]))
            self.poids.append(poids)

    def forwardPropag(self, imageMatrice):
        # on transforme la matrice de pixels en vecteur, car le réseau ne peut pas lire une image carrée
        pix = imageMatrice.reshape(-1)
        activation = [pix]
        zs = [] # pour écrire les résultats intermédiaires juste avant d'appliquer la fonction d'activation => utile pour la backward pour corriger les erreurs

        for poids in self.poids:
            z = np.dot(pix, poids) # on multiplie chaque valeur de gris de l'image par les poids
            pix = self.fonctionActivation(z)
            activation.append(pix)

        return activation[-1]

    def backPropag(self, imageMatrice, label):
        activations, zs = self.forwardPropag(imageMatrice)
        cible = np.zeros(10)
        cible[label] = 1

        deltas = [None] * len(self.poids)  # liste des deltas (une par couche de poids)

        deltas[-1] = (activations[-1] - cible) * self.deriveeActivation(zs[-1]) #delta de la couche de sortie

        for l in reversed(range(len(self.poids) - 1)):
            deltas[l] = np.dot(self.poids[l + 1], deltas[l + 1]) * self.deriveeActivation(zs[l])

        for l in range(len(self.poids)): # ici on met à jour les poids
            a = activations[l].reshape(-1, 1)  # activation de la couche l
            d = deltas[l].reshape(1, -1)  # delta de la couche l+1
            self.poids[l] -= self.learning_rate * np.dot(a, d)

class MnistDataloader(object):

    def __init__(self): # training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        # à changer en fonction de vos chemins d'accès sur vos ordinateurs
        input_path = "/Users/julineamiot/PycharmProjects/PythonProjectReseauNeurones"
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

    # on parcourt toutes les images du test
    print("Parcours de toutes les images du test...")
    for i, image in enumerate(x_test):
        sortie = reseau.forwardPropag(image)[-1]  # vecteur de 10 valeurs
        prediction = int(sortie > 0)  # neurone le plus activé
        # on affiche seulement les 5 premières images pour ne pas en avoir trop
        if i < 5:
            print(f"Image {i}, nombre réel = {y_test[i]}, prédiction réseau = {prediction}")