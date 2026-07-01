from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/StudentsPrepared.xlsx")
TARGET_COLUMN = "Target"
BINARY_TARGET_COLUMN = "Evasao"
DROPOUT_LABEL = "Desistente"

GRADE_COLUMNS = [
    "UnidadesCurriculares1SemestreGrau",
    "UnidadesCurriculares2SemestreGrau",
]

BOOLEAN_COLUMNS = [
    "NecessidadesEspeciais",
    "Devedor",
    "MensalidadesEmDia",
    "Bolsista",
    "International",
]


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Carrega a base de estudantes a partir da planilha."""
    return pd.read_excel(Path(path))


def fix_grade_value(value: float) -> float:
    """Corrige notas com magnitude incompatível com a escala academica 0-20."""
    if pd.isna(value):
        return value

    value = float(value)

    while abs(value) > 20:
        value /= 10

    return value


def fix_grade_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a correcao de escala nas colunas de grau."""
    df = df.copy()

    for column in GRADE_COLUMNS:
        df[column] = df[column].apply(fix_grade_value)

    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Cria o alvo binario de evasao: Desistente=1, demais situacoes=0."""
    df = df.copy()
    df[BINARY_TARGET_COLUMN] = (df[TARGET_COLUMN] == DROPOUT_LABEL).astype(int)
    return df


def prepare_dataset(path: str | Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega, trata e separa variaveis explicativas e alvo."""
    df = load_data(path)
    df = fix_grade_columns(df)
    df = add_binary_target(df)

    x = df.drop(columns=[TARGET_COLUMN, BINARY_TARGET_COLUMN])
    y = df[BINARY_TARGET_COLUMN]

    return x, y


def get_feature_groups(x: pd.DataFrame) -> dict[str, list[str]]:
    """Separa features por estrategia de pre-processamento."""
    categorical_columns = x.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = [
        column
        for column in x.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if column not in BOOLEAN_COLUMNS
    ]

    return {
        "categorical": categorical_columns,
        "numeric": numeric_columns,
        "boolean": BOOLEAN_COLUMNS,
    }
