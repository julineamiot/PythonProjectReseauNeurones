import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import perceptron
import toolz

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
            zs.append(z)
            pix = self.fonctionActivation(z)
            activation.append(pix)

        return activation

    def backPropag(self, imageMatrice, label):
        activations, zs = self.forwardPropag(imageMatrice)
        #cible = np.zeros(10)
        #cible[label] = 1

        # cible pour un seul neurone de sortie
        X = 0
        cible = 1 if label == X else -1  # 1 si c'est le chiffre X, sinon -1

        deltas = [None] * len(self.poids)  # liste des deltas (une par couche de poids)

        deltas[-1] = (activations[-1] - cible) * self.deriveeActivation(zs[-1]) #delta de la couche de sortie

        #on fait la backward pour les couches cachées
        for l in reversed(range(len(self.poids) - 1)):
            deltas[l] = np.dot(self.poids[l + 1], deltas[l + 1]) * self.deriveeActivation(zs[l])

        #on met à jour les poids
        for l in range(len(self.poids)):
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
    for i, image in enumerate(x_test[:10]): #test sur 10 images
        activations, zs = reseau.forwardPropag(image) #on récupère activations et valeurs avant activation
        sortie = activations[-1] #dernière couche (np.array)
        prediction = int(sortie[0] > 0) #on convertit en 0 ou 1
        # on affiche seulement les 5 premières images pour ne pas en avoir trop
        print(f"Image {i}, nombre réel = {y_test[i]}, prédiction réseau = {prediction}")

    for i, image in enumerate(x_train[:10]):  # exemple sur 10 images
        reseau.backPropag(image, y_train[i])
        print(f"Image {i} traitée via backprop")


    #taux de réussite de prédiction du réseau
    X = 7  #chiffre que le réseau doit détecter
    correct = 0  #on compte le nb de bonnes prédictions
    for image, label in zip(x_test, y_test):
        activations, zs = reseau.forwardPropag(image)
        sortie = activations[-1][0]
        prediction = 1 if sortie > 0 else 0  #prédiction du réseau
        cible = 1 if label == X else 0  #on compare la prédiction avec la vrai valeur

        if prediction == cible:
            correct += 1

    taux_reussite = correct / len(x_test) * 100
    print(f"Taux de réussite du réseau pour détecter le chiffre {X} : {taux_reussite:.2f}%")