import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os
from config import THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS, OUTPUT_PATH

def optimize_threshold(model, X_test, y_test):
    print("\n🎯 Otimizando threshold de classificação...")

    os.makedirs("outputs", exist_ok=True)

    # tentar usar probabilidades
    if hasattr(model, "predict_proba"):
        scores_raw = model.predict_proba(X_test)[:, 1]

    else:
        # fallback para modelos como LinearSVC
        scores_raw = model.decision_function(X_test)

        # normalizar para intervalo 0-1
        scores_raw = (scores_raw - scores_raw.min()) / (scores_raw.max() - scores_raw.min())

    thresholds = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)

    scores = []

    y_binary = (y_test == "Positive").astype(int)

    for t in thresholds:

        preds = (scores_raw >= t).astype(int)

        score = f1_score(y_binary, preds)

        scores.append(score)

    best_idx = np.argmax(scores)

    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    print(f"🏆 Melhor threshold: {best_threshold:.3f}")
    print(f"🏆 Melhor F1: {best_score:.4f}")

    # gráfico
    plt.figure(figsize=(7,5))

    plt.plot(thresholds, scores)
    plt.axvline(best_threshold, linestyle="--")

    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold Optimization")

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_PATH}threshold_optimization.png")
    plt.close()

    return best_threshold