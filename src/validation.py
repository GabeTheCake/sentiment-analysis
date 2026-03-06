from sklearn.model_selection import cross_val_score
from config import CV_FOLDS

def cross_validate_model(model, X, y):
    print("\n🔬 Rodando Cross Validation...")

    scores = cross_val_score(
        model,
        X,
        y,
        cv=CV_FOLDS,
        scoring="f1_weighted",
        n_jobs=-1
    )

    print(f"F1 scores: {scores}")
    print(f"F1 médio: {scores.mean():.4f}")
    print(f"Desvio padrão: {scores.std():.4f}")

    return {
        "cv_mean_f1": scores.mean(),
        "cv_std_f1": scores.std()
    }