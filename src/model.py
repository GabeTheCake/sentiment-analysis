from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from config import MAX_FEATURES, NGRAM_RANGE

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Naive Bayes": MultinomialNB(),

        "Linear SVM": LinearSVC(
            class_weight="balanced"
        )
    }

    best_model = None
    best_score = 0
    best_name = None

    for name, model in models.items():
        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=MAX_FEATURES,
                    ngram_range=NGRAM_RANGE,
                    min_df=2
                )
            ),
            ("classifier", model)
        ])

        print(f"\n🔎 Treinando {name}...")

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        score = f1_score(y_test, preds, average="weighted")
        print(f"F1 Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = pipeline
            best_name = name

    print(f"\n🏆 Melhor modelo: {best_name} ({best_score:.4f})")
    return best_model