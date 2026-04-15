<p align="center">
  <img src="logo-Infnet.png" alt="Instituto Infnet" width="300">
</p>

<h1 align="center">Operacionalização de modelos com MLOps</h1>

<p align="center">
  Projeto da disciplina <strong>Fundamentos de Machine Learning com scikit-learn</strong> — visão de engenharia: pipelines reprodutíveis, MLflow, inferência e monitoramento.
</p>

<p align="center">
  <a href="https://github.com/ubyss/Infnet-Operacionalizacao-de-modelos-com-MLOps">Repositório no GitHub</a>
</p>

---

## Sobre o projeto

Classificação binária (benigno / maligno) no contexto de **câncer de mama**, com:

- ingestão e perfil de dados (`data_prep.py`);
- experimentos comparativos **Random Forest** com `StandardScaler`, **PCA** e **LDA**, validação cruzada e `GridSearchCV` dentro de `Pipeline` (`train.py`);
- registro no **MLflow** (parâmetros, métricas, modelo);
- exploração **t-SNE** (`tsne_explore.py`);
- avaliação em holdout (`evaluate.py`);
- **API** FastAPI (`serve.py`) e **interface** Streamlit (`streamlit_app.py`);
- testes de smoke e **CI** (GitHub Actions).

O relatório técnico e as decisões estão em [`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md). O notebook [`projeto_mlops_breast_cancer.ipynb`](projeto_mlops_breast_cancer.ipynb) orquestra o fluxo completo e inclui **gráficos** (matriz de confusão, ROC, t-SNE, importâncias, distribuição de classes) e opção de subir o Streamlit a partir de uma célula.

## Requisitos

- Python 3.10+ (recomendado 3.11)
- Dependências em [`requirements.txt`](requirements.txt)

## Instalação

```bash
git clone https://github.com/ubyss/Infnet-Operacionalizacao-de-modelos-com-MLOps.git
cd Infnet-Operacionalizacao-de-modelos-com-MLOps
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso rápido

Na raiz do repositório:

```bash
python data_prep.py
python train.py
python evaluate.py
python tsne_explore.py
```

Interface web:

```bash
streamlit run streamlit_app.py
uvicorn serve:app --host 127.0.0.1 --port 8000
```

Explorar experimentos:

```bash
mlflow ui
```

Testes:

```bash
pytest tests -q
```

## Estrutura principal

| Caminho | Função |
|--------|--------|
| `data_prep.py` | Dados, splits, perfil, drift KS |
| `train.py` | Treino, MLflow, registro do modelo |
| `evaluate.py` | Métricas no conjunto de teste |
| `model_io.py` | Carrega modelo versionado |
| `serve.py` | API de inferência |
| `streamlit_app.py` | Demonstração interativa |
| `.github/workflows/ci.yml` | CI com pytest |

## Dados

Fontes suportadas: CSV local, variável de ambiente `BREAST_CANCER_CSV`, pasta `data/`, Kaggle via `kagglehub` ou fallback `sklearn.datasets`. Após o primeiro `data_prep`, artefatos ficam em `artifacts/` (gerados localmente; não versionados).

## Licença e instituição

Material acadêmico — **Instituto Infnet**. Uso do repositório conforme política da disciplina e da instituição.
