"""Instanciation d'un modèle entrainé"""
from keras.models import load_model

PATH_MODEL = "./p7_global/models"


def get_trained_model(
    model_name: str = "Embedding_custom_RNN.keras"
    ):
    """Renvoie le modèle préentrainé"""
    path = f'{PATH_MODEL}/{model_name}'
    model = load_model(path)

    return model
