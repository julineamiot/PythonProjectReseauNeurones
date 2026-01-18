import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image

#nbNeuronesCouche = [784, 32, 1] #3 couches, 1ere couche 784 neurones, (2e couche 64 neurones), 3e couche 32 neurones, 4e couche 1 neurone car doit dire si c'est un x ou pas
#X = 3 # chiffre que le programme doit apprendre/reconnaitre

nbNeuronesCouche = []

class ReseauNeurones:
    def __init__(self, nbNeuronesCouche):
        self.tailles = nbNeuronesCouche
        self.nbCouches = len(nbNeuronesCouche)
        self.poids = []
        self.biais = []
        self.learning_rate = 0.001

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
            poids = np.random.randn(self.tailles[i], self.tailles[i + 1])
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

    def backPropag(self, imageMatrice, label,X):
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
        #input_path = "/Users/julineamiot/PycharmProjects/PythonProjectReseauNeurones"
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

    mnist_dataloader = MnistDataloader()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # variations nb de couches
    for y in range(9):
        nbCouches = y + 2

        if nbCouches == 2:
            liste = [784, 1]  # nb de neuronnes s'il y a qu'une couche
        else:
            valeurs_intermediaires = [random.randint(2, 783) for _ in range(nbCouches - 2)]
            valeurs_intermediaires = sorted(valeurs_intermediaires, reverse=True)  # on trie dans l'odre décroissant
            liste = [784] + valeurs_intermediaires + [1]
        nbNeuronesCouche.append(liste)


        # variations chiffres
        for i in range(10):
            X = i

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
            reseau = ReseauNeurones(liste)
            reseau.initialiserPoids()

            # entraînement
            print("Entraînement du réseau")
            for i in range(30):
                for image, label in zip(x_train[:1000], y_train[:1000]):
                    reseau.backPropag(image, label,X)

            # test
            print("Test du réseau")
            correct = 0
            TauxVraiPositif = 0
            TauxFauxPositif = 0
            TauxFauxNégatif = 0
            TauxVraiNégatif = 0

            for image, label in zip(x_test, y_test):
                resultat = reseau.forwardPropag(image)
                activations = resultat[0]
                sortie = activations[-1][0]

                prediction = 1 if sortie > 0.5 else 0
                cible = 1 if label == X else 0

                if prediction == 1 and cible == 1:
                    TauxVraiPositif += 1
                elif prediction == 1 and cible == 0:
                    TauxFauxPositif += 1
                elif prediction == 0 and cible == 1:
                    TauxFauxNégatif += 1
                elif prediction == 0 and cible == 0:
                    TauxVraiNégatif += 1

            precision = TauxVraiPositif / (TauxVraiPositif + TauxFauxPositif) if (TauxVraiPositif + TauxFauxPositif) > 0 else 0
            rappel = TauxVraiPositif / (TauxVraiPositif + TauxFauxNégatif) if (TauxVraiPositif + TauxFauxNégatif) > 0 else 0
            f1_score = 2 * (precision * rappel) / (precision + rappel) if (precision + rappel) > 0 else 0
            tauxReussite = (TauxVraiPositif + TauxVraiNégatif) / len(x_test) * 100  # exact

            print("Taux de réussite pour détecter le chiffre " + str(X) + " avec un nombre de couches de "+ str(nbCouches) + " : " + str(tauxReussite) +"%")



"""C:\Users\Utilisateur\AppData\Local\Programs\Python\Python313\python.exe "C:\Users\Utilisateur\PycharmProjects\PythonProjectReseauNeurones\test efficacité.py" 
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 0 avec un nombre de couches de 2 : 95.28%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 1 avec un nombre de couches de 2 : 95.66%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 2 avec un nombre de couches de 2 : 91.2%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 3 avec un nombre de couches de 2 : 91.42%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 4 avec un nombre de couches de 2 : 93.30000000000001%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 5 avec un nombre de couches de 2 : 87.53999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 6 avec un nombre de couches de 2 : 94.13%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 7 avec un nombre de couches de 2 : 94.45%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 8 avec un nombre de couches de 2 : 88.33%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 9 avec un nombre de couches de 2 : 88.6%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 0 avec un nombre de couches de 3 : 95.86%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 1 avec un nombre de couches de 3 : 96.81%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 2 avec un nombre de couches de 3 : 94.16%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 3 avec un nombre de couches de 3 : 93.37%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 4 avec un nombre de couches de 3 : 93.66%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 5 avec un nombre de couches de 3 : 92.42%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 6 avec un nombre de couches de 3 : 95.39999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 7 avec un nombre de couches de 3 : 94.14%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 8 avec un nombre de couches de 3 : 92.81%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 9 avec un nombre de couches de 3 : 90.16999999999999%
C:\Users\Utilisateur\PycharmProjects\PythonProjectReseauNeurones\test efficacité.py:142: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  plt.figure(figsize=(30,20))
Entraînement du réseau
C:\Users\Utilisateur\PycharmProjects\PythonProjectReseauNeurones\test efficacité.py:29: RuntimeWarning: overflow encountered in exp
  return 1 / (1 + np.exp(-x))
Test du réseau
Taux de réussite pour détecter le chiffre 0 avec un nombre de couches de 4 : 99.08%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 1 avec un nombre de couches de 4 : 99.48%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 2 avec un nombre de couches de 4 : 97.7%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 3 avec un nombre de couches de 4 : 97.21%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 4 avec un nombre de couches de 4 : 98.07000000000001%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 5 avec un nombre de couches de 4 : 97.05%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 6 avec un nombre de couches de 4 : 97.72999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 7 avec un nombre de couches de 4 : 97.74000000000001%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 8 avec un nombre de couches de 4 : 96.02000000000001%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 9 avec un nombre de couches de 4 : 96.63000000000001%
Entraînement du réseau
C:\Users\Utilisateur\PycharmProjects\PythonProjectReseauNeurones\test efficacité.py:55: RuntimeWarning: overflow encountered in dot
  z = np.dot(pix, self.poids[i]) + self.biais[i]
C:\Users\Utilisateur\PycharmProjects\PythonProjectReseauNeurones\test efficacité.py:55: RuntimeWarning: invalid value encountered in dot
  z = np.dot(pix, self.poids[i]) + self.biais[i]
Test du réseau
Taux de réussite pour détecter le chiffre 0 avec un nombre de couches de 5 : 90.2%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 1 avec un nombre de couches de 5 : 88.64999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 2 avec un nombre de couches de 5 : 89.68%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 3 avec un nombre de couches de 5 : 89.9%
Entraînement du réseau
C:\Users\Utilisateur\PycharmProjects\PythonProjectReseauNeurones\test efficacité.py:79: RuntimeWarning: overflow encountered in dot
  deltas[l] = np.dot(deltas[l + 1], self.poids[l + 1].T) * self.ReLuPrime(zs[l])
C:\Users\Utilisateur\PycharmProjects\PythonProjectReseauNeurones\test efficacité.py:79: RuntimeWarning: invalid value encountered in multiply
  deltas[l] = np.dot(deltas[l + 1], self.poids[l + 1].T) * self.ReLuPrime(zs[l])
C:\Users\Utilisateur\PycharmProjects\PythonProjectReseauNeurones\test efficacité.py:85: RuntimeWarning: invalid value encountered in dot
  self.poids[l] = self.poids[l] - self.learning_rate * np.dot(a, d)
C:\Users\Utilisateur\PycharmProjects\PythonProjectReseauNeurones\test efficacité.py:85: RuntimeWarning: overflow encountered in dot
  self.poids[l] = self.poids[l] - self.learning_rate * np.dot(a, d)
Test du réseau
Taux de réussite pour détecter le chiffre 4 avec un nombre de couches de 5 : 90.18%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 5 avec un nombre de couches de 5 : 91.08000000000001%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 6 avec un nombre de couches de 5 : 90.42%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 7 avec un nombre de couches de 5 : 89.72%
Entraînement du réseau
C:\Users\Utilisateur\PycharmProjects\PythonProjectReseauNeurones\test efficacité.py:79: RuntimeWarning: invalid value encountered in dot
  deltas[l] = np.dot(deltas[l + 1], self.poids[l + 1].T) * self.ReLuPrime(zs[l])
Test du réseau
Taux de réussite pour détecter le chiffre 8 avec un nombre de couches de 5 : 90.25999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 9 avec un nombre de couches de 5 : 89.91%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 0 avec un nombre de couches de 6 : 90.2%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 1 avec un nombre de couches de 6 : 88.64999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 2 avec un nombre de couches de 6 : 89.68%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 3 avec un nombre de couches de 6 : 89.9%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 4 avec un nombre de couches de 6 : 90.18%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 5 avec un nombre de couches de 6 : 91.08000000000001%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 6 avec un nombre de couches de 6 : 90.42%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 7 avec un nombre de couches de 6 : 89.72%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 8 avec un nombre de couches de 6 : 90.25999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 9 avec un nombre de couches de 6 : 89.91%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 0 avec un nombre de couches de 7 : 90.2%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 1 avec un nombre de couches de 7 : 88.64999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 2 avec un nombre de couches de 7 : 89.68%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 3 avec un nombre de couches de 7 : 89.9%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 4 avec un nombre de couches de 7 : 90.18%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 5 avec un nombre de couches de 7 : 91.08000000000001%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 6 avec un nombre de couches de 7 : 90.42%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 7 avec un nombre de couches de 7 : 89.72%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 8 avec un nombre de couches de 7 : 90.25999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 9 avec un nombre de couches de 7 : 89.91%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 0 avec un nombre de couches de 8 : 90.2%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 1 avec un nombre de couches de 8 : 88.64999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 2 avec un nombre de couches de 8 : 89.68%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 3 avec un nombre de couches de 8 : 89.9%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 4 avec un nombre de couches de 8 : 90.18%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 5 avec un nombre de couches de 8 : 91.08000000000001%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 6 avec un nombre de couches de 8 : 90.42%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 7 avec un nombre de couches de 8 : 89.72%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 8 avec un nombre de couches de 8 : 90.25999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 9 avec un nombre de couches de 8 : 89.91%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 0 avec un nombre de couches de 9 : 90.2%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 1 avec un nombre de couches de 9 : 88.64999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 2 avec un nombre de couches de 9 : 89.68%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 3 avec un nombre de couches de 9 : 89.9%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 4 avec un nombre de couches de 9 : 90.18%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 5 avec un nombre de couches de 9 : 91.08000000000001%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 6 avec un nombre de couches de 9 : 90.42%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 7 avec un nombre de couches de 9 : 89.72%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 8 avec un nombre de couches de 9 : 90.25999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 9 avec un nombre de couches de 9 : 89.91%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 0 avec un nombre de couches de 10 : 90.2%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 1 avec un nombre de couches de 10 : 88.64999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 2 avec un nombre de couches de 10 : 89.68%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 3 avec un nombre de couches de 10 : 89.9%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 4 avec un nombre de couches de 10 : 90.18%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 5 avec un nombre de couches de 10 : 91.08000000000001%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 6 avec un nombre de couches de 10 : 90.42%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 7 avec un nombre de couches de 10 : 89.72%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 8 avec un nombre de couches de 10 : 90.25999999999999%
Entraînement du réseau
Test du réseau
Taux de réussite pour détecter le chiffre 9 avec un nombre de couches de 10 : 89.91%

Process finished with exit code 0
"""