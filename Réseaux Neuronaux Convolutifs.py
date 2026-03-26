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
        image = Img.open("l'image qu'on veut")
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

    def convolution(self, liste_image_pad, liste_filtre):
        '''
        :param liste_image: [matrice_R, matrice_V, matrice_B]
        :param liste_filtre: Liste de filtres (chaque filtre est une matrice 3x3)
        :return: resultats_filtres
        '''
        resultats_tous_filtres = []
        h, l = liste_image_pad[0].shape

        for filtre in liste_filtre:
            matrice_sortie_filtre = np.zeros((h - 2, l - 2))

            for i in range(h - 2):
                for j in range(l - 2):
                    pixel_final = 0
                    for canal_couleur in liste_image_pad:
                        zone = canal_couleur[i:i + 3, j:j + 3]
                        pixel_final = pixel_final + np.sum(zone * filtre)

                    matrice_sortie_filtre[i, j] = pixel_final

            resultats_tous_filtres.append(matrice_sortie_filtre)
        return resultats_tous_filtres

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

            matrice_position_max = np.zeros((nouveau_h, nouveau_l))

            for j in range(nouveau_h):
                for i in range(nouveau_l):
                    # j*taille[0] pour sauter de 2 en 2 pour ne pas chevaucher
                    zone_pooling = matrice_relu[j * taille[0]: (j + 1) * taille[0], i * taille[1]: (i + 1) * taille[1]]
                    nouvelle_matrice[j, i] = np.max(zone_pooling)

                    indice_plat = np.argmax(zone_pooling) #chaque coordonnée est nomée de 1 à 4
                    coordonnées = np.unravel_index(indice_plat, zone_pooling.shape) #on trouve sa position au sein de la matrice (2x2) sur laquelle on applique le pooling
                    position_hauteur_du_max = j * taille[0] + coordonnées[0] #on trouve sa position au sein de la matrice
                    position_largeur_du_max = i * taille[1] + coordonnées[1]
                    matrice_position_max[position_hauteur_du_max,position_largeur_du_max] = 1

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

class Backward():
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


#début de main
if __name__ == "__main__":
    mon_reseau = Convolution()

    img_rgb = mon_reseau.separation_couleurs("image")#quon devra ouvrir)
    img_pad = mon_reseau.padding(img_rgb, 1)
    images_filtrees = mon_reseau.convolution(img_pad, mes_filtres)
    images_activees = mon_reseau.relu_convolution(images_filtrees)
    # ainsi de suite jusqu'au softmax