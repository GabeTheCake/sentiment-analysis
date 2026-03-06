import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import MODEL_PATH, OUTPUT_PATH
from src.local_explanations import explain_text_prediction
from src.preprocessing import carregar_dados, clean_text

st.set_page_config(
    page_title="Plataforma de IA - Análise de Sentimentos",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource(show_spinner=False)
def carregar_modelo():
    return joblib.load(MODEL_PATH)

model = carregar_modelo()

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

@st.cache_data(show_spinner=False)
def carregar_dataset():
    df = carregar_dados()
    return df

df = carregar_dataset()

st.title("🧠 Plataforma de IA para Análise de Sentimentos")

st.markdown(
"""
Sistema de **Machine Learning para análise automática de sentimentos em textos**.

Este projeto demonstra um **pipeline completo de Data Science**, incluindo:

- NLP com TF-IDF
- Treinamento de modelos
- otimização de hiperparâmetros
- interpretabilidade com Explainable AI
"""
)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard",
    "🔍 Analisar Texto",
    "🧠 Interpretabilidade"
])

with tab1:

    st.subheader("📊 Visão Geral do Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total de textos",
            len(df)
        )

    with col2:
        st.metric(
            "Features TF-IDF",
            vectorizer.max_features
        )

    with col3:
        st.metric(
            "Modelo",
            type(classifier).__name__
        )

    st.divider()

    st.subheader("Distribuição de Sentimentos")

    sentiment_counts = df["Sentiment"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        sentiment_counts.index,
        sentiment_counts.values
    )

    ax.set_title("Distribuição do Dataset")
    ax.set_ylabel("Quantidade")

    st.pyplot(fig)

    st.divider()

    st.subheader("Exemplo do Dataset")

    st.dataframe(df.head())

with tab2:

    st.subheader("✍ Digite um texto para análise")

    texto_usuario = st.text_area(
        "Texto:",
        height=150
    )

    if st.button("Analisar Sentimento 🚀"):

        if texto_usuario.strip() == "":
            st.warning("Digite algum texto.")
        else:
            texto_limpo = clean_text(texto_usuario)
            pred = model.predict([texto_limpo])[0]

            classes = classifier.classes_

            proba = model.predict_proba([texto_limpo])[0]
            classes = model.classes_

            prob = np.max(proba)

            if str(pred).lower() in ["positive", "positivo", "pos"]:
                sentimento = "Positivo 😊"
            else:
                sentimento = "Negativo 😠"

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Sentimento",
                    sentimento
                )

            with col2:
                st.metric(
                    "Confiança",
                    f"{prob:.2f}"
                )

            st.divider()

            st.subheader("📊 Confiança do Modelo")

            fig, ax = plt.subplots(figsize=(5,3))

            bars = ax.bar(
                [str(c) for c in classes],
                proba
            )

            ax.set_ylabel("Probabilidade")
            ax.set_ylim(0,1)
            ax.set_title("Confiança do Modelo")

            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.02,
                    f"{height:.2f}",
                    ha='center'
                )

            st.pyplot(fig)

            st.divider()

            st.subheader("🔍 Explicação da decisão")

            background_data = df["Text"].sample(min(100,len(df)), random_state=42).tolist()

            explain_text_prediction(
                model,
                vectorizer,
                texto_usuario,
                background_data
            )

            try:
                st.image(f"{OUTPUT_PATH}local_explanation.png")
            except:
                st.warning("Explicação não encontrada.")


with tab3:

    st.subheader("🧠 Interpretabilidade do Modelo")

    st.markdown(
    """
    Esta seção mostra como o modelo toma decisões.

    Utilizamos técnicas de **Explainable AI (XAI)** para identificar
    quais palavras mais influenciam a classificação.
    """
    )

    st.divider()

    st.subheader("Importância Global das Palavras")

    try:
        st.image("outputs/top_positive_words.png")
        st.image("outputs/top_negative_words.png")
    except:
        st.warning("Execute o pipeline para gerar os gráficos.")

    st.divider()

    st.subheader("SHAP Summary")

    try:
        st.image(f"{OUTPUT_PATH}shap_summary.png")
    except:
        st.warning("Gráfico SHAP não encontrado.")

st.divider()

st.caption(
"Projeto de Machine Learning com Python • Scikit-learn • SHAP • Streamlit"
)