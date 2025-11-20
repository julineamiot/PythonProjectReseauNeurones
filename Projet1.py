import numpy as np
from PIL import Image

class ReseauNeurones:
    def __init__(self):
        #on initalise tous les paramètres du réseau de neurones (les poids  w, les biais b)
        self.w = None
        self.biais = None

    def ouvrirImage(self):
        #Cette fonction ouvre une image et la convertit en matrice numpy,  avec des niveaux de gris entre 0 et 255.
        image = Image.open("nom de l'image")
        imageGris = image.convert("L")
        imageMatrice = np.asarray(imageGris)
        return imageMatrice

    def forwardPropag(self, imageMatrice):
        #cette fonction calcule la sortie du réseau, ie donne 0 ou 1. Elle  trasnforme la matrice de pixels en une prédiction.
        #elle combine les pixels avec les poids pour donner un score, et  renvoie 0 ou 1 selon le score "
        a = imageMatrice.shape
        for i in range(a[0,0]):
            for j in range(a[0,1]):

        pass

    def backPropag(self, image_matrice, classif):
        #cette fonction met à jour les poids du réseau en fonction de l'erreur. Elle prend en entrée l'image et la classification initale, donc 0 ou 1.
        pass

    def main(self): #fonction principale
        #elle gère l'entraînement et les tests du réseau de neurones. Elle appelle forward et backward pour plusieurs images différentes
        pass