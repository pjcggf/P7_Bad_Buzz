"""Module pour embedding des phrases ou corpus."""
import os
from gensim.models import KeyedVectors
from gensim import downloader
import numpy as np

# On fixe le nom du dictionnaire d'embedding mais on pourra par la suite
# l'inclure comme variable de la fonction return_keyed_vectors
# ou en params de l'appli pour laisser le choix à l'utilisateur
CHOOSED_WV = 'glove-twitter-100'
WV_SAVED_NAME = 'glove_twitter'
model_info = downloader.info()['models'][CHOOSED_WV]
VECTOR_SIZE = model_info['parameters']['dimension']
VOCAB_SIZE = model_info['num_records']


def return_keyed_vectors():
    """
    Renvoie le dict pour réaliser l'embedding
    """
    path = os.path.join(".", "p7_global", "data", "keyedvectors")
    try:
        # Chargement du modèle déjà présent
        choosen_wv = KeyedVectors.load(os.path.join(path, WV_SAVED_NAME))
        vocab_size = len(choosen_wv.key_to_index)
        # Vérification de la taille du dictionnaire
        assert vocab_size == VOCAB_SIZE, \
            f"Le dictionnaire de vecteurs devrait être de taille {VOCAB_SIZE} ({vocab_size} actuellement)."
        # Vérification de la taille des embeddings
        assert choosen_wv.vector_size == VECTOR_SIZE, \
            f"L'embedding devrait être de taille {VECTOR_SIZE} ({choosen_wv.vector_size} actuellement)."

    except FileNotFoundError:
        # Chargement du dict en ligne si absent
        choosen_wv = downloader.load(CHOOSED_WV)
        # Sauvegarde du modèle
        try:
            path = os.path.join(".", "p7_global", "data", "keyedvectors")
            os.mkdir(path)
        except FileExistsError:
            pass
        choosen_wv.save(os.path.join(path, WV_SAVED_NAME))
        vocab_size = len(choosen_wv.key_to_index)
        # Vérification de la taille du dictionnaire
        assert vocab_size == VOCAB_SIZE, \
            f"Le dictionnaire de vecteurs devrait être de taille {VOCAB_SIZE} ({vocab_size} actuellement)."
        # Vérification de la taille des embeddings
        assert choosen_wv.vector_size == VECTOR_SIZE, \
            f"L'embedding devrait être de taille {VECTOR_SIZE} ({choosen_wv.vector_size} actuellement)."

    return choosen_wv


def embed_sentence(wv, sentence):
    """
    Fonction pour convertir un tweet en une matrice (aka tableau de vecteurs) contenant les représentations
    vectorielle de chaque mots dans l'espace de plongement
    """
    embedded_sentence = []
    for word in sentence:
        # On vérifie si le mot fait bien partie du dictionnaire final
        # (peut-être exclue car trop ou pas assez fréquent).
        if word in wv:
            embedded_sentence.append(wv[word])

    return np.array(embedded_sentence)
