# 💬 Análise de Sentimentos em Redes Sociais com NLP e Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)  
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)  
![NLP](https://img.shields.io/badge/NLP-TFIDF%20%7C%20Text%20Classification-purple)  
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-red)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Projeto de **Processamento de Linguagem Natural (NLP)** focado em **classificação automática de sentimentos em textos curtos de redes sociais**.

O projeto implementa um **pipeline completo de Machine Learning**, incluindo:

- pré-processamento de texto
- vetorização com TF-IDF
- treinamento e comparação de modelos
- otimização de threshold
- interpretabilidade do modelo
- detecção de data drift
- geração de visualizações
- aplicação interativa com **Streamlit**

Este projeto demonstra habilidades em **NLP, modelagem supervisionada, arquitetura modular de projetos em Python e interpretabilidade de modelos de Machine Learning**.

---

# 🎯 Problema

Plataformas digitais geram diariamente grandes volumes de texto:

- redes sociais  
- avaliações de produtos  
- comentários de clientes  
- feedback de usuários  

Analisar manualmente esse volume de informação é inviável.

**Pergunta central:**

> É possível classificar automaticamente o sentimento de textos utilizando técnicas de NLP e Machine Learning?

---

# 🧠 Técnicas Utilizadas

O projeto inclui diversas etapas comuns em pipelines profissionais de NLP:

### Processamento de Texto

- limpeza de texto
- normalização
- tokenização

### Vetorização

- **TF-IDF (Term Frequency – Inverse Document Frequency)**

### Modelos de Machine Learning

- Logistic Regression
- Naive Bayes
- Linear Support Vector Machine (Linear SVM)

### Avaliação de Modelos

- F1-Score
- Matriz de confusão
- comparação entre modelos

### Interpretabilidade

- análise de palavras mais importantes
- visualização de coeficientes do modelo
- explicações locais

### Monitoramento

- detecção de **data drift**

### Interface Interativa

- aplicação web com **Streamlit** para análise de textos em tempo real

---

# 🛠️ Tecnologias e Bibliotecas

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SHAP
- Streamlit

---

# 📁 Estrutura do Projeto

```
sentiment-analysis/
│
├── data/
│   └── sentimentdataset.csv
│
├── models/
│   └── sentiment_model.pkl
│
├── notebook/
│   └── Sentiment_analysis.ipynb
│
├── output/
│   ├── confusion_matrix.png
│   ├── drift_report.txt
│   ├── local_explanation.png
│   ├── metrics.json
│   ├── shap_summary.png
│   ├── threshold_optimization.png
│   ├── top_negative_words.png
│   └── top_positive_words.png
│
├── src/
│   ├── drift_detection.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── hyperparameter_tunning.py
│   ├── interpretability.py
│   ├── local_explanations.py
│   ├── modeling.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── threshold_optimization.py
│   └── validation.py
│
├── app.py
├── config.py
├── main.py
├── README.md
├── requirements.txt
└── LICENSE.txt
```

---

# ▶️ Como Executar o Projeto

### 1️⃣ Clonar o repositório

```bash
git clone https://github.com/seu-usuario/sentiment-analysis.git
cd sentiment-analysis
```

---

### 2️⃣ Criar ambiente virtual

```bash
python -m venv venv
```

---

### 3️⃣ Ativar ambiente

Windows:

```
venv\Scripts\activate
```

Linux / Mac:

```
source venv/bin/activate
```

---

### 4️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

---

# 🚀 Executar o Pipeline Completo

Para executar todo o pipeline de treinamento:

```bash
python main.py
```

O pipeline executa automaticamente:

- carregamento dos dados
- pré-processamento de texto
- vetorização com TF-IDF
- treinamento de múltiplos modelos
- avaliação comparativa
- otimização de threshold
- geração de métricas
- geração de gráficos
- análise de interpretabilidade
- detecção de data drift
- salvamento do modelo final

---

# 🌐 Aplicação Interativa

O projeto inclui uma **interface web interativa com Streamlit**, que permite analisar sentimentos de textos em tempo real.

Para executar:

```bash
streamlit run app.py
```

A aplicação permite:

- inserir textos manualmente
- visualizar classificação de sentimento
- visualizar confiança da previsão

---

# 📊 Outputs Gerados

Durante a execução do pipeline, o projeto gera diversos artefatos:

- **confusion_matrix.png** → matriz de confusão do modelo  
- **top_positive_words.png** → palavras mais associadas a sentimento positivo  
- **top_negative_words.png** → palavras mais associadas a sentimento negativo  
- **shap_summary.png** → análise de interpretabilidade global  
- **local_explanation.png** → explicação local de previsões  
- **threshold_optimization.png** → otimização do threshold de classificação  
- **metrics.json** → métricas do modelo  
- **drift_report.txt** → relatório de detecção de drift  

---

# 📚 Notebook da Análise

A análise exploratória e explicação detalhada do processo está disponível em:

```
notebook/Sentiment_analysis.ipynb
```

O notebook inclui:

- exploração do dataset
- preparação dos dados
- vetorização de texto
- comparação de modelos
- avaliação
- interpretação do modelo

---

# 🚀 Possíveis Melhorias Futuras

- utilização de **modelos baseados em Transformers (BERT)**
- implementação de **monitoramento contínuo de drift**
- deploy do modelo como **API (FastAPI)**
- integração com **dashboard de monitoramento**
- treinamento com datasets maiores

---

# 👤 Autor

Gabriel  
Data Analytics | Data Science | Machine Learning  

Interesse em oportunidades **remotas e internacionais**.

---

# 📄 Licença

Este projeto está sob a licença **MIT**.