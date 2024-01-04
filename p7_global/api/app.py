"""Module contenant les endpoints de l'API"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tensorflow import expand_dims
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Opening JSON file
with open('./p7_global/data/tokenizer_custom.json', 'r') as openfile:
    json_object = json.load(openfile)
    tokenizer_loaded = tokenizer_from_json(json_object)

from p7_global.ml_logic.model import get_trained_model
from p7_global.ml_logic.embedding import return_keyed_vectors, embed_sentence
from p7_global.ml_logic.data import cleaning_text_lemma

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# Instancie le modèle et le précharge en mémoire dès le démarrage de l'instance
# pour éviter des temps de chargements trop longs lors de la requete.
# Idem pour le tokenizer custom pré-entrainé
app.state.model = get_trained_model()

with open('./p7_global/data/tokenizer_custom.json', 'r', encoding='utf-8') as openfile:
    json_object = json.load(openfile)
    app.state.tokenizer = tokenizer_from_json(json_object)

@app.get('/')
def welcome_home():
    """
    Message d'accueil à la racine
    """
    print('Welcome to root')
    return 'Welcome to root'

@app.get('/predict_single_tweet')
def predict_single_tweet(text :str):
    """
    Renvoie une prédiction binaire sur le sentiment associé à un texte.
    """
    # Récupération du modèle pré-instancié
    model = app.state.model
    # Récupération du dict d'embedding pré-instancié
    tokenizer = app.state.tokenizer
    # Nettoyage et lemmatization de l'input
    text = cleaning_text_lemma(text)
    # Vectorisation du texte
    text = tokenizer.texts_to_sequences([text])
    # Padding du tenseur pour ramener la taille de nb_mots_vectorisé à 40
    text = pad_sequences(text, dtype='float16', padding='post', maxlen=50)



    try:
        # On réalise la prédiction sur le texte vectorisé
        pred = model(text)
        result = ""
        # Si on onbtient un score supérieur à 0.6 on considère le tweet
        # comme positif.
        if pred.numpy() > 0.6:
            result = 'Positif'
        else:
            result = 'Négatif'
    except Exception:
        result = "Prédiction erronnée"

    return result


if __name__ == "__main__":
    app.run()
