import numpy as np
from PIL import Image

class ReseauNeurones:
    def __init__(self):
        self.nbCouches = 2
        self.nbNeuronesCouche = [None, 32, 1]  # entrée, couche cachée, sortie
        self.donneesEntree = None
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
        self.poidsEntree = np.random.random_sample(taille_image, self.nbNeuronesCouche[1])
        self.poidsSortie = np.random.random_sample(self.nbNeuronesCouche[1], 1)

    def forwardPropag(self, imageMatrice):
        #cette fonction calcule la sortie du réseau, ie donne 0 ou 1. Elle  trasnforme la matrice de pixels en une prédiction.
        #elle combine les pixels avec les poids pour donner un score, et  renvoie 0 ou 1 selon le score "
        pix = [imageMatrice[i][j] for i in range(imageMatrice.shape[0]) for j in range(imageMatrice.shape[1])]
        taillepix = len(pix)
        matPoids = np.random.random_sample((taillepix, self.nbNeuronesCouche))
        
        for i in pix:
            i = i * 

        pass

    def backPropag(self, image_matrice, classif):
        "cette fonction met à jour les poids du réseau en fonction de l'erreur. Elle prend en entrée l'image et la classification"
        "initale, donc 0 ou 1."
        pass

    def main(self):
        "elle gère l'entraînement et les tests du réseau de neurones. Elle appelle forward et backward pour plusieurs images différentes "
        pass

"""mat = np.arange(6).reshape(2, 3)
print(mat)
pix = [mat[i][j] for i in range(mat.shape[0]) for j in range(mat.shape[1])]
print(pix)"""

"""le porgramme doit orendre en entrée une matrcie numpy de taille paramétrable. Le réseau doit avir une couche d'entrée de la tille de l'image et à la fin on a une sortie. Pour chaque couche, faire produit matriciel entre les indices de la matrice avec poids. Si valeur du neurone inféireur à 0, on renvoie -1, sinon on renvoie 1"""