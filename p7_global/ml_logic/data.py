"""Module de récupération et préparation des données pour entrainement
de modèle."""
import re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from sklearn.model_selection import train_test_split

def compress(df):
    """
    Reduces size of dataframe by downcasting numerical columns
    """
    input_size = df.memory_usage(index=True).sum() / 1024
    print("new dataframe size: ", round(input_size, 2), 'kB')

    in_size = df.memory_usage(index=True).sum()
    for types in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=types))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=types)
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100

    print(f"optimized size by {round(ratio, 2)}")
    print("new dataframe size: ", round(out_size / 1024, 2), " kB")

    return df


# Regex de nettoyage
# TEXT_CLEANING_RE = '@\S+|https\S+|http\S+|&.*;|\d+|[^a-zA-Z]'
TEXT_CLEANING_RE = '@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+'
# Dict des stop words
try:
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")



def cleaning_text_stem(text: str, exclude_stop_words: bool=True) -> list:
    """Retourne une phrase en liste nettoyée ou non des stop words
        après stemming"""
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

    tokens = []
    if exclude_stop_words:
        for token in text.split():
            if token not in stop_words:
                tokens.append(stemmer.stem(token))
    else:
        for token in text.split():
            tokens.append(stemmer.stem(token))

    return tokens


def cleaning_text_lemma(text: str,
                        exclude_stop_words: bool = True,
                        pos:str = 'v') -> list:
    """Retourne une phrase en liste nettoyée ou non des stop words
        après lemmatization"""

    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

    if exclude_stop_words:
        tokens = [token for token in text.split() if token not in stop_words]
    else:
        tokens = [token for token in text.split()]

    if pos in ['v', 'n']:
        tokens = [WordNetLemmatizer().lemmatize(token, pos=pos)
                  for token in tokens]
    else:
        raise AttributeError(
            f'Le param `pos` ne peut pas prendre la valeur {pos}, uniquement `v` ou `n`.')

    return tokens


def import_data() -> pd.DataFrame:
    """
    Renvoie le dataset complet (~1.6 m tweets) dédoublonné non traité
    et sans `user`, `date` et `flag`.
    """
    dataset_columns = ["target", "ids", "date", "flag", "user", "text"]
    http_link = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+7%C2%A0-+D%C3%A9tectez+les+Bad+Buzz+gr%C3%A2ce+au+Deep+Learning/sentiment140.zip"
    df = pd.read_csv(http_link,
                    encoding='latin',
                    header=None,
                     names=dataset_columns)
    df.drop(['user', 'date', 'flag'], axis=1, inplace=True)
    df = df[~df.ids.duplicated(keep=False)].reset_index(drop=True)

    return compress(df)


def get_train_and_test_set(df: pd.DataFrame,
                           test_size: float = 0.2,
                           sample: bool = False,
                           sample_size: int = 4000,
                           lemmatizing: bool = True,
                           pos: str = 'v',
                           exclude_stop_words: bool = True) -> np.array:
    """
    Renvoie les jeux de données d'entrainement et de test traités (X_train, X_test, y_train, y_test)
    """
    if sample:
        df = df.sample(sample_size, random_state=32)

    if lemmatizing:
        df['text_norm'] = df.text.apply(
            cleaning_text_lemma, pos=pos, exclude_stop_words=exclude_stop_words)

    else:
        df['text_norm'] = df.text.apply(
            cleaning_text_stem, exclude_stop_words=exclude_stop_words)

    df_train, df_test = train_test_split(df,
                                         test_size=test_size,
                                         random_state=32)
    X_train = df_train.text_norm
    X_test = df_test.text_norm
    y_train = df_train.target.apply(lambda x: 1 if x == 4 else 0).to_numpy()
    y_test = df_test.target.apply(lambda x: 1 if x == 4 else 0).to_numpy()

    return X_train, X_test, y_train, y_test


def get_train_val_and_test_set(df: pd.DataFrame,
                               test_size: float = 0.2,
                               sample: bool = False,
                               sample_size: int = 4000,
                               lemmatizing: bool = True,
                               pos: str = 'v',
                               exclude_stop_words: bool = True) -> np.array:
    """
    Renvoie les jeux de données d'entrainement, de validation et de test traités
    (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if sample:
        df = df.sample(sample_size, random_state=32)

    if lemmatizing:
        df['text_norm'] = df.text.apply(
            cleaning_text_lemma, pos=pos, exclude_stop_words=exclude_stop_words)

    else:
        df['text_norm'] = df.text.apply(
            cleaning_text_stem, exclude_stop_words=exclude_stop_words)

    df_train, df_test = train_test_split(df,
                                         test_size=test_size,
                                         random_state=32)

    df_train, df_val = train_test_split(df_train,
                                        test_size=test_size,
                                        random_state=32)
    X_train = df_train.text_norm
    X_val = df_val.text_norm
    X_test = df_test.text_norm
    y_train = df_train.target.apply(lambda x: 1 if x == 4 else 0).to_numpy()
    y_val = df_val.target.apply(lambda x: 1 if x == 4 else 0).to_numpy()
    y_test = df_test.target.apply(lambda x: 1 if x == 4 else 0).to_numpy()

    return X_train, X_val, X_test, y_train, y_val, y_test
