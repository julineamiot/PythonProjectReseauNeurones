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
carte de saillance : dire que l'on a beaucoup utiliser pour détecter l'objet
elements modifiés pendant la backwardpropag : les poids et les biais, à la fois ceux de la convolution et ceux des couches fully connected """

import os
import struct
from array import array
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as Img

class ForwardCNN():
    def __init__(self, epaisseur_padding=1, taille_pooling=(2,2)):
        self.epaisseur_padding = epaisseur_padding
        self.taille_pooling = taille_pooling
        self.filtres = [(1/2) * np.random.randn(3,3) for i in range(4)]

    def separation_couleurs(self, image): #henri
        '''
        :param image: une matrice qui va etre divisee en 3 matrices selon la couleur (RBV)
        :return: une liste de matrices selon la couleur
        '''
        ref_img_r, ref_img_g, ref_img_b = image.split()
        matrice_r = np.array(ref_img_r) / 255 # division pour normaliser
        matrice_g = np.array(ref_img_g) / 255
        matrice_b = np.array(ref_img_b) / 255
        liste_couleurs = [matrice_r, matrice_g, matrice_b]
        return liste_couleurs


    def padding(self, liste_image): #juline
        '''
        :param liste_image: les 3 matrices 28x28 de chaque couleur
        :param epaisseur: nb de lignes et de colonnes a rajouter pour chaque matrice
        :return: liste de 3 matrices + grandes
        '''
        liste_resultat = []
        for img in liste_image:
            h, w = img.shape
            img_padding = np.zeros((h + 2 * self.epaisseur_padding, w + 2 * self.epaisseur_padding))
            e = self.epaisseur_padding
            img_padding[e: h + e, e : w + e] = img
            liste_resultat.append(img_padding)
        return liste_resultat

    def convolution(self, liste_image_pad):
        '''
        :param liste_image_pad: [R, G, B] en 30x30
        :param liste_filtre: liste de filtres (chaque filtre est 3x3)
        '''
        resultats_tous_filtres = []
        h, l = liste_image_pad[0].shape

        for filtre in self.filtres:
            matrice_sortie_filtre = np.zeros((h - 2, l - 2))

            for ligne in range(h - 2):
                for col in range(l - 2):
                    pixel_final = 0
                    for canal_couleur in liste_image_pad:
                        zone = canal_couleur[ligne : ligne + 3, col : col + 3]
                        pixel_final = pixel_final + np.sum(zone * filtre)

                    matrice_sortie_filtre[ligne, col] = pixel_final

            resultats_tous_filtres.append(matrice_sortie_filtre)
        return resultats_tous_filtres

    """
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
    """

    def leaky_relu_convolution(self, a, liste_matrice_convo):
        '''
        :param liste_matrice_convo: liste de matrices de sortie de la fonction convolution
        si une valeur de la matrice est négative, on met un 0, sinon la valeur reste comme elle est
        a : appartient [0;1[ mais très inférieur à 1
        :return: liste de matrices de meme taille avec des a*valeur si la valeur < 0 et valeurs si valeur > 0
        '''
        liste_relu = []
        for matrice in liste_matrice_convo:
            z = np.where(matrice > 0, matrice, a*matrice)
            liste_relu.append(z)
        return liste_relu

    def max_pooling(self, liste_relu):
        '''
        :param matrice_relu: matrice apres activation
        :param taille: dimension de la matrice de selection pour le pooling (souvent 2x2, mais on generalise)
        :return: matrice de taille plus petite avec max des 4 pixels pour chaque selection
        '''

        liste_matrice_reduite = []
        hauteur, largeur = self.taille_pooling # hauteurs et largeurs de la matrice de pool

        for matrice_relu in liste_relu:
            h_avant, l_avant = matrice_relu.shape

            if h_avant % 2 == 0 and l_avant % 2 == 0:
                h_apres = h_avant // 2
                l_apres = l_avant // 2

                nouvelle_matrice = np.zeros((h_apres, l_apres))

                for ligne in range(h_apres):
                    for col in range(l_apres):
                        depart_h, fin_h = 2*ligne, 2*(ligne + 1)
                        depart_l, fin_l = 2*col, 2*(col + 1)
                        zone_pooling = matrice_relu[depart_h: fin_h, depart_l: fin_l]
                        nouvelle_matrice[ligne, col] = np.max(zone_pooling)

                liste_matrice_reduite.append(nouvelle_matrice)

            else:
                np.insert(matrice_relu, l_avant + 1, 0, axis=1)
                np.insert(matrice_relu, h_avant + 1, 0, axis=0)

                h_apres = (h_avant + 1) // 2
                l_apres = (l_avant + 1) // 2


                nouvelle_matrice = np.zeros((h_apres, l_apres))

                for ligne in range(h_apres):
                    for col in range(l_apres):
                        depart_h, fin_h = 2*ligne, 2*(ligne + 1)
                        depart_l, fin_l = 2*col, 2*(col + 1)
                        zone_pooling = matrice_relu[depart_h : fin_h, depart_l: fin_l]
                        nouvelle_matrice[ligne, col] = np.max(zone_pooling)

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

class BackwardCNN():
    def __init__(self):
        return None

    def back_dense(self):
        return None

    def unfmatten(self): #backward applatir
        return None

    def back_pooling(self):
        return None

    def convolution(self):
        return None



class ReseauNeurones():
    def __init__(self, tailles):
        self.tailles = tailles
        self.poids = []
        self.biais = []
        self.learning_rate = 0.005
        for i in range(len(tailles) - 1):
            self.poids.append(np.random.randn(tailles[i], tailles[i + 1]))
            self.biais.append(np.zeros(tailles[i + 1]))

    def softmax_final(self, scores): #juline
        '''
        :param scores: sortie de la couche dense
        on transforme les scores en probabilités pour chaque classe
        :return: vecteur de 10 probabilités
        '''
        exp_scores = np.exp(scores)
        probas = exp_scores / np.sum(exp_scores)
        return probas

    def forwardPropag(self, vecteur_entree): #henri
        '''
        :param vecteur_entree: sortie de la fonction applatir
        on fait la somme pondérée (produit scalaire + biais)
        :return: vecteur de scores
        '''
        activation=[vecteur_entree]
        zs=[]
        pix=vecteur_entree

        for i in range(len(self.poids)):
            z=np.dot(pix, self.poids[i]) + self.biais[i]
            zs.append(z)

            if i == len (self.poids) - 1:
                pix = self.softmax_final(z)
            else:
                pix=np.maximum(0,z)
            activation.append(pix)
        return activation, zs

    def backwardPropag(self, vecteur_entree, label):
        activation, zs = self.forwardPropag(vecteur_entree)

        cible=np.zeros(self.tailles[-1])
        cible[label]=1
        deltas=[None] * len(self.poids)
        deltas[-1]=activation[-1] - cible

        for l in reversed(range(len(self.poids)-1)):
            relu_prime=np.where(zs[l]<0,0,1)
            deltas[l]=np.dot(deltas[l+1], self.poids[l+1].T) * relu_prime

        for l in range(len(self.poids)):
            a = activation[l].reshape(-1, 1)
            d = deltas[l].reshape(1, -1)
            self.poids[l] = self.poids[l] - self.learning_rate * np.dot(a, d)
            self.biais[l] = self.biais[l] - self.learning_rate * deltas[l]



class CatDogDataloader(object):

    def __init__(self): # training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        #self.input_path = "/Users/julineamiot/Documents/PycharmProjects/PythonProjectReseauNeurones/PetImages"
        self.input_path = r"C:\Users\Utilisateur\PycharmProjects\PythonProjectReseauNeurones\PetImages"
        self.train_path = os.path.join(self.input_path, "train")
        self.test_path = os.path.join(self.input_path, "test")

    def read_images_labels(self, dossier_cible, nb_images_max=500):
        """
        Remplace l'ancienne lecture binaire par une lecture de fichiers JPG
        """
        images = []
        labels = []
        classes = {'cats': 0, 'dogs': 1}

        for nom_classe, etiquette in classes.items():
            chemin_classe = os.path.join(dossier_cible, nom_classe)
            fichiers = os.listdir(chemin_classe)[:nb_images_max]

            for f in fichiers:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(chemin_classe, f)
                    img = Img.open(img_path).convert('RGB').resize((28, 28))
                    images.append(img)
                    labels.append(etiquette)

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.train_path, nb_images_max=100)
        x_test, y_test = self.read_images_labels(self.test_path, nb_images_max=20)
        return (x_train, y_train), (x_test, y_test)



#début de main
if __name__ == "__main__":
    dataloader = CatDogDataloader()
    (x_train, y_train), (x_test, y_test) = dataloader.load_data()

    reseau_cnn = ForwardCNN(epaisseur_padding=1, taille_pooling=(2, 2))
    tailles = [784, 64, 2]
    reseau_simple = ReseauNeurones(tailles)
    indices = list(range(len(x_train)))
    random.shuffle(indices)
    x_train_melange = []
    y_train_melange = []
    for i in indices:
        x_train_melange.append(x_train[i])
        y_train_melange.append(y_train[i])
    x_train = x_train_melange
    y_train = y_train_melange

    # entrainement
    for i in range(3):
        for image, label in zip(x_train, y_train):
            # forward
            # partie convolution
            rgb = reseau_cnn.separation_couleurs(image)
            padding = reseau_cnn.padding(rgb)
            convolu = reseau_cnn.convolution(padding)
            activation = reseau_cnn.leaky_relu_convolution(0.001, convolu)
            pooling = reseau_cnn.max_pooling(activation)
            vecteur_final = reseau_cnn.applatir(pooling)

            # partie reseau de neurones simple
            resultat = reseau_simple.backwardPropag(vecteur_final, label)

    # test
    reussite=0
    for image, label in zip(x_test, y_test):
        activation_test = reseau_cnn.leaky_relu_convolution(0.001, reseau_cnn.convolution(reseau_cnn.padding(reseau_cnn.separation_couleurs(image))))
        vecteur_test = reseau_cnn.applatir(reseau_cnn.max_pooling(activation_test))
        resultats, _ =reseau_simple.forwardPropag(vecteur_test)
        prediction = np.argmax(resultats[-1])

        if prediction == label:
            reussite= reussite+1

    print ("Taux de reussite final : ", (reussite/len(x_test))*100, "%")