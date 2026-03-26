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
from PIL import Image as Img

class Convolution():
    def __init__(self):
        return None

    def separation_couleurs(self, image): #henri
        '''
        :param image: une matrice qui va etre divisee en 3 matrices selon la couleur (RBV)
        :return: une liste de matrices selon la couleur
        '''
        ref_img_r, ref_img_g, ref_img_b = image.split()
        matrice_r = np.array(ref_img_r)
        matrice_g = np.array(ref_img_g)
        matrice_b = np.array(ref_img_b)
        liste_couleurs = [matrice_r, matrice_g, matrice_b]
        return liste_couleurs


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

    def convolution(self, liste_image, liste_filtre):
        '''
        :param liste_image: [matrice_R, matrice_V, matrice_B]
        :param liste_filtre: Liste de filtres (chaque filtre est une matrice 3x3)
        '''
        resultats_filtres = []
        h, l = liste_image[0].shape  #taille de l'image

        for filtre in liste_filtre:
            matrice_sortie = np.zeros((h - 2 , l - 2)) # matrice vide pour stocker le résultat du filtre
            for i in range(h - 2):
                for j in range(l - 2):
                    somme_canaux = 0
                    # On fait le calcul pour chaque couleur (R, V, B)
                    for canal in liste_image:
                        # On découpe la zone 3x3
                        zone = canal[i:i + 3, j:j + 3]
                        # Multiplication et somme
                        somme_canaux += np.sum(zone * filtre)

                    # On enregistre le résultat final pour ce pixel
                    matrice_sortie[i, j] = somme_canaux

            resultats_filtres.append(matrice_sortie)

        return resultats_filtres

    def relu_convolution(self, liste_matrice_convo): #juline
        '''
        :param liste_matrice_convo: liste de matrices de sortie de la fonction convolution
        si une valeur de la matrice est négative, on met un 0, sinon la valeur reste comme elle est
        :return: liste de matrices de meme taille avec des 0 et des valeurs positive
        '''
        liste_relu = []
        for matrice in liste_matrice_convo:
            z = np.maximum(0, matrice)
            liste_relu.append(z)
        return liste_relu

    def max_pooling(self, liste_relu, taille): #henri
        '''
        :param matrice_relu: matrice apres activation
        :param taille: dimension de la matrice de selection pour le pooling (souvent 2x2, mais on generalise)
        :return: matrice de taille plus petite avec max des 4 pixels pour chaque selection
        '''
        liste_matrice_reduite = []
        for matrice_relu in liste_relu:
            h, l = matrice_relu.shape

            nouveau_h = h // taille[0]
            nouveau_l = l // taille[1]

            nouvelle_matrice = np.zeros((nouveau_h, nouveau_l))

            for j in range(nouveau_h):
                for i in range(nouveau_l):
                    # j*taille[0] pour sauter de 2 en 2 pour ne pas chevaucher
                    zone_pooling = matrice_relu[j * taille[0]: (j + 1) * taille[0], i * taille[1]: (i + 1) * taille[1]]
                    nouvelle_matrice[j, i] = np.max(zone_pooling)

            liste_matrice_reduite.append(nouvelle_matrice)
        return liste_matrice_reduite

    def applatir(self, liste_matrice): #juline
        '''
        :param liste_matrice: images apres le dernier pooling
        prendre tous les chiffres de toutes les matrices et les mettre à la suite dans un seul tableau 1D
        :return: un vecteur une dimension avec toutes les valeurs
        '''
        vecteur_apla = []
        for matrice in liste_matrice:
            ligne = matrice.flatten()
            vecteur_apla.extend(ligne) # pour ajouter éléments par éléments, pas la liste entière
            x = np.array(vecteur_apla) # conversion array pour les fonctions suivantes
        return x

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
        exp_scores = np.exp(scores)
        probas = exp_scores / np.sum(exp_scores)
        return probas



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
    reseau = None
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