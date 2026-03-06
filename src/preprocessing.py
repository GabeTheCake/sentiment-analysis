from pathlib import Path
import pandas as pd
from config import DATA_PATH
import re

def carregar_dados():
    caminho = Path(DATA_PATH)
    if not caminho.exists():
        raise FileNotFoundError(f"O arquivo {caminho} não foi encontrado.")

    df = pd.read_csv(caminho)

    df = df[["Text", "Sentiment"]]

    positive_keywords = [
        "positive","joy","excitement","contentment","gratitude","happy","hopeful",
        "awe","euphoria","admiration","adoration","affection","amusement","blessed",
        "calmness","celebration","compassion","confidence","delight","elation",
        "enthusiasm","gratified","inspiration","love","optimism","pride","relief",
        "serenity","wonder","accomplishment","appreciation"
    ]

    negative_keywords = [
        "negative","anger","anxiety","despair","loneliness","grief","sad",
        "embarrassed","confusion","fear","frustration","disappointment",
        "bitterness","betrayal","boredom","guilt","hate","hopeless","hurt",
        "insecurity","jealousy","loss","melancholy","pain","regret",
        "rejection","stress","tension","worry","pressure","obstacle"
    ]

    def map_sentiment(sentiment):
        sentiment = sentiment.lower()

        for word in positive_keywords:
            if word in sentiment:
                return "Positive"
        for word in negative_keywords:
            if word in sentiment:
                return "Negative"
            

    df["Sentiment"] = df["Sentiment"].apply(map_sentiment)
    df = df.dropna()

    return df

def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text

def preprocess_data(df):

    df = df.copy()

    df["Text"] = df["Text"].apply(clean_text)

    X = df["Text"]
    y = df["Sentiment"]

    return X, y