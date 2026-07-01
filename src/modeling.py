from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(feature_groups: dict[str, list[str]]) -> ColumnTransformer:
    """Cria o pre-processador para categoricas, numericas e binarias."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_groups["categorical"]),
            ("num", StandardScaler(), feature_groups["numeric"]),
            ("bool", "passthrough", feature_groups["boolean"]),
        ]
    )


def build_model(feature_groups: dict[str, list[str]]) -> Pipeline:
    """Cria o pipeline final de pre-processamento e classificacao."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_groups)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
