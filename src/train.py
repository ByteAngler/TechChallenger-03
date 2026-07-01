import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data_processing import DATA_PATH, get_feature_groups, prepare_dataset
from src.modeling import build_model


MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("reports/metrics.json")


def evaluate_test_set(model, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def evaluate_cross_validation(model, x_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = cross_validate(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )

    metrics = {}
    for metric in scoring:
        metrics[metric] = {
            "train_mean": results[f"train_{metric}"].mean(),
            "train_std": results[f"train_{metric}"].std(),
            "validation_mean": results[f"test_{metric}"].mean(),
            "validation_std": results[f"test_{metric}"].std(),
        }

    return metrics


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    x, y = prepare_dataset(DATA_PATH)
    feature_groups = get_feature_groups(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model(feature_groups)
    model.fit(x_train, y_train)

    test_metrics = evaluate_test_set(model, x_test, y_test)
    cv_metrics = evaluate_cross_validation(model, x_train, y_train)

    metrics = {
        "data_path": str(DATA_PATH),
        "target_mapping": {
            "Desistente": 1,
            "Graduado": 0,
            "Matriculado": 0,
        },
        "feature_groups": feature_groups,
        "test": test_metrics,
        "cross_validation": cv_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_json(metrics, METRICS_PATH)

    print(json.dumps(metrics, indent=2))
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Metricas salvas em: {METRICS_PATH}")


if __name__ == "__main__":
    main()
