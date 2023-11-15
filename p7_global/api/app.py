"""Module contenant les endpoints de l'API"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tensorflow import expand_dims
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
# Idem pour le dictionnaire d'embedding
app.state.model = get_trained_model()
app.state.wv = return_keyed_vectors()

@app.get('/')
def welcome_home():
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
    wv = app.state.wv
    # Nettoyage et lemmatization de l'input
    text = cleaning_text_lemma(text)
    # Vectorisation de l'input
    text = embed_sentence(wv, text)
    # Ajout d'une dimension pour correspondre à la shape attendu par le
    # modèle (1, nb_mots_vectorisés, 100)
    text = expand_dims(text, 0)
    # Padding du tenseur pour ramener la taille de nb_mots_vectorisé à 40
    text = pad_sequences(text, dtype='float16', padding='post', maxlen=40)



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
