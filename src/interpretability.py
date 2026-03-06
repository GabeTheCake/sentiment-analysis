import numpy as np
import matplotlib.pyplot as plt
import os
from config import OUTPUT_PATH

def plot_top_words(model, top_n=20):
    print("🔎 Extraindo palavras mais importantes...")

    os.makedirs("outputs", exist_ok=True)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]

    top_positive = np.argsort(coefficients)[-top_n:]
    top_negative = np.argsort(coefficients)[:top_n]

    positive_words = feature_names[top_positive]
    positive_scores = coefficients[top_positive]

    negative_words = feature_names[top_negative]
    negative_scores = coefficients[top_negative]

    # POSITIVE
    plt.figure(figsize=(8,6))
    plt.barh(positive_words, positive_scores)
    plt.title("Top Positive Words")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}top_positive_words.png")
    plt.close()

    # NEGATIVE
    plt.figure(figsize=(8,6))
    plt.barh(negative_words, negative_scores)
    plt.title("Top Negative Words")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}top_negative_words.png")
    plt.close()

    print("📊 Gráficos de palavras salvos em outputs/")