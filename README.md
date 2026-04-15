<p align="center">
  <img src="docs/img/logo-Infnet.png" alt="Instituto Infnet" width="300">
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

O relatório técnico está em [`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md). O notebook [`projeto_mlops_breast_cancer.ipynb`](projeto_mlops_breast_cancer.ipynb) orquestra o fluxo e inclui gráficos e opção de subir o Streamlit em uma célula.

## Visualizações (notebook — seção 6.1)

Painel gerado após treino e avaliação (matriz de confusão, ROC, t-SNE, importâncias do Random Forest) e distribuição de classes no treino:

<p align="center">
  <img src="docs/img/painel-modelo-prod.png" alt="Painel do modelo em produção" width="720"><br>
  <em>Painel do modelo em produção (holdout + t-SNE + importâncias)</em>
</p>

<p align="center">
  <img src="docs/img/classes.png" alt="Distribuição de classes no treino" width="480"><br>
  <em>Distribuição de classes no conjunto de treino</em>
</p>

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

## Estrutura do repositório

```
├── docs/img/              # Logo Infnet e figuras do README
├── tests/                 # Pytest (CI)
├── projeto_mlops_breast_cancer.ipynb
├── data_prep.py           # Dados, splits, perfil, drift KS
├── train.py               # Treino + MLflow + registro
├── evaluate.py            # Métricas no teste
├── model_io.py            # Carrega modelo versionado
├── tsne_explore.py        # t-SNE exploratório
├── serve.py               # API FastAPI
├── streamlit_app.py       # Interface Streamlit
├── RELATORIO_TECNICO.md
├── requirements.txt
└── .github/workflows/ci.yml
```

Artefatos gerados localmente (`artifacts/`, `mlruns/`) não são versionados; são recriados ao rodar os scripts.

## Scripts Python — todos em uso

| Arquivo | Uso |
|--------|-----|
| `data_prep.py` | Notebook, `train`, drift, `tsne_explore`, `streamlit` (splits) |
| `train.py` | Notebook, CI indireto via `tests` que importa `build_pipelines` |
| `evaluate.py` | Notebook, métricas finais |
| `model_io.py` | `evaluate`, `serve`, `streamlit`, notebook (carga do modelo) |
| `tsne_explore.py` | Notebook, redução não linear |
| `serve.py` | API de inferência |
| `streamlit_app.py` | Demonstração |

Não há módulos `.py` dispensáveis: remover qualquer um quebra o fluxo do notebook, da API, do Streamlit ou dos testes.

## Dados

Fontes suportadas: CSV local, variável de ambiente `BREAST_CANCER_CSV`, pasta `data/`, Kaggle via `kagglehub` ou fallback `sklearn.datasets`. Após o primeiro `data_prep`, artefatos ficam em `artifacts/` (gerados localmente).

## Licença e instituição

Material acadêmico — **Instituto Infnet**. Uso do repositório conforme política da disciplina e da instituição.
