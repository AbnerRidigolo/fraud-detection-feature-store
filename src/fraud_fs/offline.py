"""Construção do dataset de treino a partir da OFFLINE store (point-in-time join).

Este módulo demonstra a razão de existir mais importante de uma feature store: o
**ponto-no-tempo correto** ao montar dados de treino.

Como funciona `get_historical_features`:

1. Damos a ele um *entity dataframe*: uma linha por evento que queremos rotular —
   `(account_id, event_timestamp, label, + dados do request)`.
2. Para CADA linha, o Feast faz um "as-of join": pega o valor da feature cujo
   `event_timestamp` é o MAIOR que ainda seja <= ao timestamp do evento (e dentro
   do TTL). Ou seja, "o que eu sabia sobre esta conta NAQUELE instante".
3. As on-demand features são calculadas em cima desse resultado, linha a linha.

Fazer esse join "na mão" com SQL é onde nascem os bugs de leakage mais sutis (um
`>=` no lugar de `<`, esquecer o TTL, juntar pelo timestamp errado). A feature
store encapsula isso corretamente — e essa é exatamente a dor que ela resolve.
"""

from __future__ import annotations

import pandas as pd

from fraud_fs.config import FEATURE_REFS, RAW_TRANSACTIONS, TRAINING_DATASET
from fraud_fs.store import get_store


def build_entity_df() -> pd.DataFrame:
    """Monta o entity dataframe: um evento (transação) por linha, com o rótulo.

    Inclui os campos do RequestSource (amount, lat, lon, txn_unixtime) porque a
    on-demand feature view precisa deles para calcular z-score, distância etc.
    durante o join histórico.
    """
    txns = pd.read_parquet(RAW_TRANSACTIONS)
    entity = txns[["account_id", "event_timestamp", "amount", "lat", "lon", "is_fraud"]].copy()
    entity["txn_unixtime"] = entity["event_timestamp"].apply(lambda x: x.timestamp())
    return entity


def build_training_dataset(write: bool = True) -> pd.DataFrame:
    """Gera o dataset de treino com features point-in-time-correct."""
    store = get_store()
    entity_df = build_entity_df()

    training = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()

    if write:
        TRAINING_DATASET.parent.mkdir(parents=True, exist_ok=True)
        training.to_parquet(TRAINING_DATASET, index=False)
    return training


def main() -> None:
    df = build_training_dataset(write=True)
    print(f"Dataset de treino: {len(df):,} linhas, {df['is_fraud'].sum():,} fraudes.")
    print(f"Escrito em {TRAINING_DATASET}")


if __name__ == "__main__":
    main()
