from src.preprocessing import carregar_dados, preprocess_data
from src.model import train_models
from src.evaluation import evaluate_model, plot_confusion_matrix
from src.validation import cross_validate_model
from src.interpretability import plot_top_words
from src.hyperparameter_tunning import tune_model
from src.explainability import generate_shap_explanations
from src.local_explanations import explain_text_prediction
from src.threshold_optimization import optimize_threshold
from src.drift_detection import detect_data_drift
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE, MODEL_PATH, OUTPUT_PATH
import joblib

def run_pipeline():
    print("📥 Carregando dados...")
    df = carregar_dados()

    print("🧹 Pré-processando dados...")
    X, y = preprocess_data(df)

    print("✂ Dividindo treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("🤖 Treinando múltiplos modelos...")
    model = train_models(X_train, X_test, y_train, y_test)

    print("⚙ Otimizando hiperparâmetros...")
    model = tune_model(model, X_train, y_train)

    print("🧠 Detectando Data Drift...")
    detect_data_drift(model.named_steps["tfidf"], X_train, X_test)

    print("🔬 Validando modelo com Cross Validation...")
    cv_results = cross_validate_model(model, X, y)

    print("📊 Avaliando modelo...")
    results = evaluate_model(model, X_test, y_test)

    print("📈 Plotando matriz de confusão...")
    plot_confusion_matrix(y_test, model.predict(X_test))

    print("🎯 Otimizando threshold...")
    best_threshold = optimize_threshold(model, X_test, y_test)
    print(f"🎯 Threshold ótimo encontrado: {best_threshold:.3f}")

    print("🧠 Gerando interpretabilidade do modelo...")
    plot_top_words(model)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    print("🧠 Gerando explicações com SHAP...")
    generate_shap_explanations(classifier, X_train, X_test, vectorizer )

    print("\n===== RESULTADOS =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("🔍 Gerando explicação para um exemplo de texto...")
    sample_text = "I feel joy and gratitude for my friends but sometimes life brings frustration and challenges"
    
    explain_text_prediction(
        model,
        model.named_steps["tfidf"],
        sample_text,
        X_train.sample(200)
    )

    print("💾 Salvando modelo...")
    joblib.dump(model, MODEL_PATH)

    print("\n🚀 Pipeline completo!")
    print("📁 Outputs salvos em:", OUTPUT_PATH)
    print("💾 Modelo salvo em:", MODEL_PATH)