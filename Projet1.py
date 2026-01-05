import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class ReseauNeurones:
    def __init__(self):
        self.nbCouches = 2
        self.nbNeuronesCouche = [None, 32, 1]  # entrée, couche cachée, sortie
        self.poidsEntree = None
        self.poidsSortie = None

    def fonctionActivation(self, x):
        return np.where(x < 0, -1, 1)

    def ouvrirImage(self):
        #Cette fonction ouvre une image et la convertit en matrice numpy,  avec des niveaux de gris entre 0 et 255.
        image = Image.open("nom de l'image")
        imageGris = image.convert("L")
        imageMatrice = np.asarray(imageGris)
        return imageMatrice

    def initialiserPoids(self, taille_image):
        self.nbNeuronesCouche[0] = taille_image # taille_image = nombre de pixels total
        self.poidsEntree = np.random.random_sample((taille_image, self.nbNeuronesCouche[1]))
        self.poidsSortie = np.random.random_sample((self.nbNeuronesCouche[1], 1))

    def forwardPropag(self, imageMatrice):
        # on transforme la matrice de pixels en vecteur
        pix = imageMatrice.reshape(-1)

        # produit matriciel entrée x couche cachée
        z1 = np.dot(pix, self.poidsEntree) #np.dot pour les produits np
        a1 = self.fonctionActivation(z1)

        # produit matriciel cachée x sortie
        z2 = np.dot(a1, self.poidsSortie)
        a2 = self.fonctionActivation(z2)

        return a2

    def backPropag(self, image_matrice, classif):
        "cette fonction met à jour les poids du réseau en fonction de l'erreur. Elle prend en entrée l'image et la classification"
        "initale, donc 0 ou 1."
        pass

    def main(self):
        "elle gère l'entraînement et les tests du réseau de neurones. Elle appelle forward et backward pour plusieurs images différentes "
        pass


"""le porgramme doit orendre en entrée une matrcie numpy de taille paramétrable. Le réseau doit avir une couche d'entrée de la tille de 
l'image et à la fin on a une sortie. Pour chaque couche, faire produit matriciel entre les indices de la matrice avec poids. 
Si valeur du neurone inféireur à 0, on renvoie -1, sinon on renvoie 1"""


# MNIST Data Loader Class

class MnistDataloader(object):

    def __init__(self): # training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):

        #
        # Set file paths based on added MNIST Datasets
        #
        input_path = "C:/Users/Utilisateur/OneDrive/Documents/Cours/TSE/L3/Programmation, magistère/Projet"
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
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


# Verify Reading Dataset via MnistDataloader class


# Helper function to show a list of images with their relating titles

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


# Load MINST dataset


# if __name__=="__main__":
if True:

    mnist_dataloader = MnistDataloader()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    #
    # Show some random training and test images
    #
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