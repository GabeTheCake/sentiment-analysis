from sklearn.model_selection import GridSearchCV

def tune_model(model, X_train, y_train):

    print("\n⚙ Rodando GridSearch para otimizar hiperparâmetros...")

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("🏆 Melhor parâmetro encontrado:")
    print(grid.best_params_)

    print("🏆 Melhor F1 (CV):")
    print(grid.best_score_)

    return grid.best_estimator_