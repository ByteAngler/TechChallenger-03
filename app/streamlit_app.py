from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import (
    BOOLEAN_COLUMNS,
    DATA_PATH,
    TARGET_COLUMN,
    fix_grade_columns,
    load_data,
)


MODEL_PATH = PROJECT_ROOT / "models/model.joblib"
REFERENCE_DATA_PATH = PROJECT_ROOT / DATA_PATH


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    df = load_data(REFERENCE_DATA_PATH)
    return df.drop(columns=[TARGET_COLUMN])


def build_input_form(reference_df: pd.DataFrame) -> pd.DataFrame:
    input_data = {}

    categorical_columns = reference_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = [
        column
        for column in reference_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if column not in BOOLEAN_COLUMNS
    ]

    with st.form("prediction_form"):
        st.subheader("Dados do aluno")

        st.markdown("**Dados cadastrais**")
        for column in categorical_columns:
            options = sorted(reference_df[column].dropna().unique().tolist())
            input_data[column] = st.selectbox(column, options)

        st.markdown("**Indicadores booleanos**")
        for column in BOOLEAN_COLUMNS:
            input_data[column] = st.selectbox(column, [0, 1], format_func=lambda value: "Sim" if value == 1 else "Nao")

        st.markdown("**Dados numericos**")
        for column in numeric_columns:
            default_value = float(reference_df[column].median())
            input_data[column] = st.number_input(column, value=default_value)

        submitted = st.form_submit_button("Prever evasao")

    if not submitted:
        return pd.DataFrame()

    input_df = pd.DataFrame([input_data], columns=reference_df.columns)
    input_df = fix_grade_columns(input_df)
    return input_df


def main() -> None:
    st.set_page_config(page_title="Predicao de Evasao")
    st.title("Predicao de evasao de estudantes")

    model = load_model()
    reference_df = load_reference_data()
    input_df = build_input_form(reference_df)

    if input_df.empty:
        st.info("Preencha os dados e clique em Prever evasao.")
        return

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])

    st.subheader("Resultado")
    st.metric("Probabilidade de evasao", f"{probability:.1%}")

    if prediction == 1:
        st.error("Classificacao: estudante com risco de evasao.")
    else:
        st.success("Classificacao: estudante sem indicacao de evasao.")

    st.caption("Modelo RandomForest treinado com tratamento das colunas de grau e pipeline de pre-processamento.")


if __name__ == "__main__":
    main()
