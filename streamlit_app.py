from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from data_prep import TARGET_COL, read_saved_splits
from model_io import load_model_bundle


@st.cache_resource
def cached_model_bundle():
    return load_model_bundle()


@st.cache_data
def cached_train_xy():
    got = read_saved_splits()
    if got is None:
        return None
    X_train, _, _, y_train, _, _, _ = got
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True)


def _table_from_row(row: pd.Series, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Atributo": feature_names, "Valor": [float(row[c]) for c in feature_names]}
    )


def _predict(model, X_one: pd.DataFrame) -> tuple[int, float, float]:
    t0 = time.perf_counter()
    proba = float(model.predict_proba(X_one)[0, 1])
    pred = int(model.predict(X_one)[0])
    dt_ms = (time.perf_counter() - t0) * 1000
    return pred, proba, dt_ms


st.set_page_config(
    page_title="Breast Cancer — Inferência",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Inferência — câncer de mama")
st.caption("Valores gerados a partir de casos reais do conjunto de treino (mesma distribuição do projeto).")

try:
    model, feature_names, run_id = cached_model_bundle()
    pack = cached_train_xy()
    if pack is None:
        raise FileNotFoundError("Splits ausentes. Rode data_prep.py ou o notebook.")
    X_train, y_train = pack
    X_train = X_train[feature_names]
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Modelo")
    st.code(run_id[:12] + "…", language=None)
    st.caption("0 = Benigno · 1 = Maligno")

if "feat_tbl" not in st.session_state:
    r0 = X_train.sample(1, random_state=42).iloc[0]
    st.session_state.feat_tbl = _table_from_row(r0, feature_names)
if "editor_rev" not in st.session_state:
    st.session_state.editor_rev = 0

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Gerar aleatório", use_container_width=True, type="primary"):
        r = X_train.sample(1).iloc[0]
        st.session_state.feat_tbl = _table_from_row(r, feature_names)
        st.session_state.editor_rev += 1
        st.rerun()
with c2:
    if st.button("Gerar com tendência positiva (maligno)", use_container_width=True):
        Xm = X_train[y_train == 1]
        if len(Xm) == 0:
            st.error("Sem amostras malignas no treino.")
        else:
            r = Xm.sample(1).iloc[0]
            st.session_state.feat_tbl = _table_from_row(r, feature_names)
            st.session_state.editor_rev += 1
            st.rerun()
with c3:
    if st.button("Gerar com tendência negativa (benigno)", use_container_width=True):
        Xb = X_train[y_train == 0]
        if len(Xb) == 0:
            st.error("Sem amostras benignas no treino.")
        else:
            r = Xb.sample(1).iloc[0]
            st.session_state.feat_tbl = _table_from_row(r, feature_names)
            st.session_state.editor_rev += 1
            st.rerun()

st.subheader("Atributos (editáveis)")
edited = st.data_editor(
    st.session_state.feat_tbl,
    disabled=["Atributo"],
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    key=f"feat_editor_{st.session_state.editor_rev}",
)

try:
    vals = {a: float(v) for a, v in zip(edited["Atributo"], edited["Valor"], strict=True)}
    X_one = pd.DataFrame([vals])[feature_names]
    pred, proba, dt_ms = _predict(model, X_one)
except (ValueError, TypeError, KeyError):
    st.warning("Ajuste os valores numéricos na tabela para ver a predição.")
else:
    st.divider()
    st.subheader("Resultado")
    m1, m2, m3 = st.columns(3)
    with m1:
        nome = "Maligno" if pred == 1 else "Benigno"
        st.metric("Classe prevista", nome)
    with m2:
        st.metric("Probabilidade de maligno", f"{proba:.2%}")
    with m3:
        st.metric("Latência", f"{dt_ms:.2f} ms")
    st.progress(min(1.0, max(0.0, proba)))
    st.caption("Barra = probabilidade de maligno segundo o modelo. A predição atualiza ao gerar novo caso ou ao editar a tabela.")
