import numpy as np
from scipy.spatial.distance import jensenshannon
from config import OUTPUT_PATH


def detect_data_drift(vectorizer, X_train_texts, X_new_texts):
    print("\n🧠 Verificando Data Drift...")

    X_train = vectorizer.transform(X_train_texts)
    X_new = vectorizer.transform(X_new_texts)

    train_distribution = np.asarray(X_train.mean(axis=0)).flatten()
    new_distribution = np.asarray(X_new.mean(axis=0)).flatten()

    train_distribution = train_distribution / train_distribution.sum()
    new_distribution = new_distribution / new_distribution.sum()

    drift_score = jensenshannon(train_distribution, new_distribution)

    print(f"📊 Drift score: {drift_score:.4f}")

    if drift_score > 0.1:
        print("⚠️ Data Drift detectado!")
    else:
        print("✅ Sem drift relevante.")

    with open(f"{OUTPUT_PATH}drift_report.txt", "w") as f:
        f.write(f"Drift score: {drift_score}\n")
        f.write("Threshold: 0.1\n")

    return drift_score