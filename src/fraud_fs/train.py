"""Treino do modelo de fraude consumindo features da OFFLINE store.

Decisões de modelagem e o porquê de cada uma:

* **Split TEMPORAL, não aleatório.** Fraude é um problema temporal: treinamos no
  passado para prever o futuro. Um `train_test_split` aleatório vazaria padrões
  do futuro para o treino e inflaria as métricas. Ordenamos por tempo e usamos os
  primeiros 80% para treinar e os últimos 20% para testar — como seria na vida real.

* **Métricas de classe desbalanceada.** Com ~1,5% de fraude, um modelo que diz
  "nunca é fraude" acerta 98,5% — acurácia é inútil aqui. Reportamos ROC-AUC e,
  principalmente, PR-AUC (average precision) + precision/recall, que medem o que
  importa: pegar fraude sem afogar o time de análise em falsos positivos.

* **`class_weight="balanced"`.** Faz o modelo "pesar" mais os raros casos de
  fraude, compensando o desbalanceamento sem precisar reamostrar.

* **Pipeline com imputação.** A primeira transação de cada conta tem perfil
  vazio (NaN) — *cold start*. O `SimpleImputer` trata isso de forma explícita e,
  por estar no Pipeline, a MESMA imputação é aplicada na inferência.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from fraud_fs.config import MODEL_FEATURES, MODEL_PATH, SEED, TRAINING_DATASET


def _time_split(df: pd.DataFrame, frac: float = 0.8):
    """Divide treino/teste por ordem cronológica (sem embaralhar o tempo)."""
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]


def train(write: bool = True) -> dict:
    """Treina o modelo, avalia num conjunto futuro e persiste o artefato."""
    df = pd.read_parquet(TRAINING_DATASET)
    train_df, test_df = _time_split(df)

    x_train, y_train = train_df[MODEL_FEATURES], train_df["is_fraud"]
    x_test, y_test = test_df[MODEL_FEATURES], test_df["is_fraud"]

    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "fraud_rate_test": float(y_test.mean()),
    }

    print("=== Avaliação (conjunto futuro / hold-out temporal) ===")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}   PR-AUC: {metrics['pr_auc']:.3f}")
    print(
        f"Treino: {metrics['n_train']:,} | Teste: {metrics['n_test']:,} "
        f"| Fraude no teste: {metrics['fraud_rate_test']:.2%}"
    )
    print(classification_report(y_test, preds, digits=3, zero_division=0))

    importances = sorted(
        zip(MODEL_FEATURES, model.named_steps["clf"].feature_importances_, strict=True),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("Importância das features:")
    for name, imp in importances:
        print(f"  {name:<24} {imp:.3f}")

    if write:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": model, "features": MODEL_FEATURES, "threshold": 0.5, "metrics": metrics},
            MODEL_PATH,
        )
        print(f"\nModelo salvo em {MODEL_PATH}")

    return metrics


if __name__ == "__main__":
    train(write=True)
