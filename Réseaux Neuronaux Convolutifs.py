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
pooling : réduire l'information et résumer l'information (moyenne, + grande valeur)
dense : mettre toutes les info en un veteur (si beaucoup d'image on peut faire 2 fois cette étape) le nb de dense correspond au nb de neurones que l'on doit avoir en sortie
pour éviter le sur apprentissage, on peut tourner, flouter l'image
carte de saillance : dire que l'on a beaucoup utiliser pour détecter l'objet"""

import struct
from array import array
import random
import matplotlib.pyplot as plt
import numpy as np

class Convolution():
    def __init__(self):
        return None

    def separation_couleurs(self, image): #henri
        '''
        :param image: une matrice qui va etre divisee en 3 matrices selon la couleur (RBV)
        :return: une liste de matrices ?
        '''
        return None

    def padding(self, liste_image, epaisseur): #juline
        '''
        :param liste_image: les 3 matrices 28x28 de chaque couleur
        :param epaisseur: nb de lignes et de colonnes a rajouter pour chaque matrice
        :return: liste de 3 matrices + grandes
        '''
        liste_resultat = []
        for img in liste_image:
            h, w = img.shape
            img_padding = np.zeros((h + 2 * epaisseur, w + 2 * epaisseur))
            img_padding[epaisseur: h + epaisseur, epaisseur: w + epaisseur] = img
            liste_resultat.append(img_padding)
        return liste_resultat

    def convolution(self, liste_image, liste_filtre): # henri
        taille_hauteur = 28
        taille_largeur = 28
        for y in range(taille_largeur):
            for i in range(taille_hauteur):
                x = liste_filtre * liste_image


        '''
        :param liste_image: idem
        :param liste_filtre: liste de matrices 3x3
        à chaque position on multiplie les pixels de l'image par les poids du filtre et on fait la somme
        et pour chaque matrice de la liste, on fait la somme des 3 valeurs qu'on a trouvé pour le pixel
        :return: pour chaque filtre, une matrice plus petite
        '''
        return None

    def relu_convolution(self, liste_matrice_convo): #juline
        '''
        :param liste_matrice_convo: liste de matrices de sortie de la fonction convolution
        si une valeur de la matrice est négative, on met un 0, sinon la valeur reste comme elle est
        :return: liste de matrices de meme taille avec des 0 et des valeurs positive
        '''
        for i in range(len(liste_matrice_convo)):
            z = np.where(liste_matrice_convo[i] < 0, 0, 1)
        return z

    def max_pooling(self, matrice_relu, taille): #henri
        taille_matrice_relu = len(matrice_relu)
        nb_nouvelle_matrice = taille_matrice_relu/taille
        for i in range(nb_nouvelle_matrice):
        '''
        :param matrice_relu: matrice apres activation
        :param taille: dimension de la matrice de selection pour le pooling (souvent 2x2, mais on generalise)
        :return: matrice de taille plus petite avec max des 4 pixels pour chaque selection
        '''
        return None

    def applatir(self, liste_matrice): #juline
        '''
        :param liste_matrice: images apres le dernier pooling
        prendre tous les chiffres de toutes les matrices et les mettre à la suite dans un seul tableau 1D
        :return: un vecteur une dimension avec toutes les valeurs
        '''
        for i in range(len(liste_matrice)):
            x=liste_matrice[i]
            for j in range 

    def dense_layer(self, vecteur_aplatit, poids, biais): #henri
        '''
        :param vecteur_aplatit: sortie de la fonction applatir
        :param poids: matrice de poids de la couche dense
        :param biais: vecteur de biais
        on fait la somme pondérée (produit scalaire + biais)
        :return: vecteur de scores
        '''
        score = np.dot(vecteur_aplatit,poids)+biais
        return score

    def softmax_final(self, scores): #juline
        '''
        :param scores: sortie de la couche dense
        on transforme les scores en probabilités pour chaque classe
        :return: vecteur de 10 probabilités
        '''
        return None


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