import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from config import OUTPUT_PATH

def explain_text_prediction(model, vectorizer, text, background_data, top_n=10):
    os.makedirs("outputs", exist_ok=True)

    classifier = model.named_steps["classifier"]

    # transformar dados
    X_background = vectorizer.transform(background_data).toarray()
    X_text = vectorizer.transform([text]).toarray()

    feature_names = np.array(vectorizer.get_feature_names_out())

    # criar explainer com background
    explainer = shap.LinearExplainer(classifier, X_background)

    shap_values = explainer.shap_values(X_text)[0]

    # pegar apenas palavras presentes no texto
    present_idx = np.where(X_text[0] > 0)[0]

    shap_values = shap_values[present_idx]
    words = feature_names[present_idx]

    # top features
    top_idx = np.argsort(np.abs(shap_values))[-top_n:]

    shap_values = shap_values[top_idx]
    words = words[top_idx]

    # ordenar
    order = np.argsort(shap_values)

    shap_values = shap_values[order]
    words = words[order]

    plt.figure(figsize=(8,6))

    colors = ["red" if v > 0 else "blue" for v in shap_values]

    plt.barh(words, shap_values, color=colors)

    plt.xlabel("Impact on prediction (SHAP)")
    plt.title("Local Explanation")

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_PATH}local_explanation.png")
    plt.close()

    print("📊 Explicação local salva em outputs/local_explanation.png")