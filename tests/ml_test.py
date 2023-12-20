import pytest
from gensim.models import KeyedVectors
from gensim import downloader
from p7_global.ml_logic.embedding import return_keyed_vectors

CHOOSED_WV = 'glove-twitter-100'
WV_SAVED_NAME = 'glove_twitter'
model_info = downloader.info()['models'][CHOOSED_WV]
VECTOR_SIZE = model_info['parameters']['dimension']
VOCAB_SIZE = model_info['num_records']

def check_vocab_exist():
    """
    Vérifie si le dict d'embedding est présent
    """
    return_keyed_vectors()
    try:
        choosen_wv = KeyedVectors.load(f'./p7_global/data/keyedvectors/{WV_SAVED_NAME}')
        if choosen_wv:
            return "Dict d'embedding trouvé"

    except FileNotFoundError as exc:
        raise FileNotFoundError('Fichier non trouvé') from exc

def check_vocab_size():
    """
    Vérifie la taille du dict
    """
    choosen_wv = KeyedVectors.load(f'./p7_global/data/keyedvectors/{WV_SAVED_NAME}')
    vocab_size = len(choosen_wv.key_to_index)

    assert vocab_size == VOCAB_SIZE, \
    f"Le dictionnaire de vecteurs devrait être de taille {VOCAB_SIZE} ({vocab_size} actuellement)."

def check_vector_size():
    """
    Vérifie la taille des vecteurs
    """
    choosen_wv = KeyedVectors.load(f'./p7_global/data/keyedvectors/{WV_SAVED_NAME}')
    assert choosen_wv.vector_size == VECTOR_SIZE, \
    f"L'embedding devrait être de taille {VECTOR_SIZE} ({choosen_wv.vector_size} actuellement)."

if __name__ == '__main__':
    pytest.main()
