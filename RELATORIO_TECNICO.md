# Relatório técnico — Engenharia de ML e MLOps (Breast Cancer)

## 1. Contexto e objetivos

**Problema de negócio:** apoiar o fluxo clínico de triagem/classificação de exames associados a câncer de mama, priorizando a **detecção de casos malignos** (classe positiva) para reduzir falsos negativos, sem ignorar o custo de alarmes desnecessários.

**Objetivo técnico:** construir um classificador binário reprodutível, com pipelines `scikit-learn` sem vazamento de dados, comparar abordagens (baseline, PCA, LDA), registrar tudo no **MLflow** e expor inferência versionada (API e interface).

**Critérios de sucesso alinhados à operação:**

| Dimensão | Meta |
|----------|------|
| Qualidade (validação) | F1 na validação competitivo entre abordagens; escolha explícita do melhor run |
| Generalização | Holdout final (`X_test`) só para estimativa final; CV estratificada no treino |
| Latência | Inferência pontual na ordem de milissegundos em CPU (meta de projeto: típico < 200 ms) |
| Rastreabilidade | Parâmetros, métricas e artefatos de modelo registrados no MLflow |

**Métricas de negócio (proxy):**

- **Recall (sensibilidade)** sobre maligno: fração de malignos corretamente identificados — prioridade quando o custo de perder um caso positivo é alto.
- **Precisão** sobre maligno: entre os alertados como malignos, quantos são realmente malignos — relevante para carga de revisão e confiança do sistema de apoio.
- **F1** como equilíbrio operacional quando precisão e recall precisam ser negociados.
- **AUC-ROC** como resumo de capacidade de ordenação/ranking entre classes em cenários com limiar ajustável.

Essas métricas foram registradas no MLflow (validação) e consolidadas no holdout (`evaluate.py`).

## 2. Dados e engenharia

**Fontes:** CSV local, variável `BREAST_CANCER_CSV`, `data/breast-cancer.csv`, dataset Kaggle (`kagglehub`) ou fallback `sklearn.datasets.load_breast_cancer`, com **mesmo esquema de features** para comparabilidade.

**Qualidade:** relatório em `artifacts/data_profile/data_profile.json` (ausências, outliers por regra IQR, pares altamente correlacionados, distribuição de classes).

**Viés e limitações:** desbalanceamento leve entre classes; splits **estratificados**; o dataset é **estático e histórico** — não representa necessariamente drift futuro nem todos os centros/populações.

**Pipelines:** `StandardScaler` e reduções ficam **dentro** do `Pipeline` e o `GridSearchCV` usa apenas `X_train`, evitando vazamento do pré-processamento para validação cruzada.

## 3. Experimentação (MLflow)

**Experimentos comparativos:**

1. `rf_standard_scaler` — Random Forest com escalonamento.
2. `rf_pca` — RF com PCA (variância explicada como hiperparâmetro efetivo via `n_components` em fração).
3. `rf_lda` — RF após LDA supervisionado (redução de dimensão discriminativa).

**Método:** `StratifiedKFold` + `GridSearchCV` com métrica de seleção interna F1; tempos de treino (`train_grid_seconds`) e métricas de validação (`val_*`) logados por run.

**Seleção do modelo candidato à produção:** o run com maior **`val_f1`** entre as três abordagens; registro no Model Registry (`BreastCancerClassifier`) e metadados em `artifacts/registry_selection.json`.

**Interpretação crítica:** RF oferece bom padrão de baseline; PCA controla dimensionalidade e custo em espaço latente; LDA usa rótulos na redução (útil se a separação linear entre classes for forte). A escolha final equilibra desempenho na validação, custo de treino e complexidade operacional (pipeline único serializável).

## 4. Redução de dimensionalidade

| Técnica | Papel no projeto | Serviço |
|---------|------------------|--------|
| PCA | Não supervisionada; compressão por variância | Integrada ao pipeline se vencer a seleção |
| LDA | Supervisionada; eixos discriminantes | Integrada ao pipeline se vencer a seleção |
| t-SNE | Visualização/exploração da estrutura local em 2D | **Fora** do pipeline de inferência (não reprodutível entre amostras da mesma forma que PCA/LDA para scoring) |

Artefatos t-SNE: `artifacts/exploration/tsne_train_2d.csv` e `tsne_meta.json` (gerados por `tsne_explore.py` e pelo notebook).

**Trade-offs:** PCA/LDA alteram interpretabilidade direta por feature original; t-SNE ajuda narrativa e auditoria exploratória, mas não substitui métricas no holdout nem o contrato de API.

## 5. Avaliação final e modelo em produção

**Holdout:** `evaluate.py` grava `artifacts/evaluation/test_metrics.json` e relatório de classificação — única leitura honesta de generalização após fixar o modelo.

**Empacotamento:** modelo servido via `mlflow.sklearn.load_model` a partir do `run_id` selecionado; mesma ordem de features que `artifacts/splits/feature_names.json`.

## 6. Operacionalização

- **API:** FastAPI (`serve.py`) — `/health`, `/predict` com latência medida.
- **UI:** Streamlit (`streamlit_app.py`) para demonstração interativa.
- **Versionamento:** MLflow tracking + Model Registry; `registry_selection.json` documenta a escolha.

## 7. Monitoramento, drift e métricas

**Técnicas:** latência por requisição, distribuição de probabilidade de maligno, acurácia/precisão/recall/F1/AUC em janelas (quando houver rótulos).

**Negócio:** taxa de casos elevados pelo modelo, taxa de confirmação clínica (quando disponível), tempo médio até decisão — dependem de integração hospitalar; aqui definimos **o que** monitorar e **como** mapear para as métricas técnicas.

**Drift de dados:** teste **Kolmogorov–Smirnov** univariado entre referência (treino) e amostra nova ou holdout (`data_drift_ks_report`), com relatório em `artifacts/drift/drift_report.json`. Indica mudança em **P(X)**; drift conceitual **P(Y|X)** exige rótulos e validação clínica periódica.

**MLflow:** runs históricos permitem comparar degradação de métricas entre retreinos.

## 8. Re-treinamento contínuo (estratégia)

1. **Gatilhos:** queda de F1/recall em validação temporal; aumento de flags de drift KS; volume mínimo de novos dados rotulados.
2. **Processo:** reexecutar `data_prep` (perfil atualizado) → `train` (novos runs) → comparar no MLflow → promover versão no Registry → smoke tests (`pytest`, inferência em API).
3. **Frequência:** mensal ou trimestral conforme disponibilidade de rótulos; hotfix se drift severo em features críticas.

## 9. Integração contínua (CI)

O repositório inclui workflow GitHub Actions (`.github/workflows/ci.yml`) que instala dependências e executa testes automatizados em branches `main`/`master` e em pull requests, garantindo que imports e verificações básicas passem antes de integrar mudanças.

## 10. Como reproduzir

1. `pip install -r requirements.txt`
2. `python data_prep.py` (ou notebook, seção 1) — atalho na raiz; implementação em `breast_cancer_mlops/data_prep.py`
3. `python train.py` — idem `breast_cancer_mlops/train.py`
4. `python evaluate.py`
5. `python tsne_explore.py` (exploração t-SNE)
6. `uvicorn serve:app` e/ou `streamlit run streamlit_app.py` (scripts finos na raiz importam o pacote)
7. `mlflow ui` apontando para a pasta `mlruns/`

Equivalente: `python -m breast_cancer_mlops.train` (e outros módulos), a partir da raiz do repositório.
