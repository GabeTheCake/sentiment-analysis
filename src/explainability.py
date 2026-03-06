import shap
import matplotlib.pyplot as plt
import os
from config import OUTPUT_PATH

def generate_shap_explanations(classifier, X_train, X_test, vectorizer):
    os.makedirs("outputs", exist_ok=True)

    X_train_vec = vectorizer.transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    feature_names = vectorizer.get_feature_names_out()

    explainer = shap.LinearExplainer(classifier, X_train_vec)

    shap_values = explainer.shap_values(X_test_vec)

    shap.summary_plot(
        shap_values,
        X_test_vec,
        feature_names=feature_names,
        show=False
    )

    plt.savefig(f"{OUTPUT_PATH}shap_summary.png", bbox_inches="tight")
    plt.close()

    print("📊 SHAP summary salvo em outputs/")