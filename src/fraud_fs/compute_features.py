"""Cálculo das features de perfil do cliente — a tabela que alimenta a offline store.

POR QUE este passo existe separado da geração: o Feast (na sua forma OSS) NÃO
computa features para você. Ele é o sistema de *armazenamento, versionamento e
serving* das features; a engenharia de feature em si é responsabilidade de um
pipeline a montante (aqui, este arquivo). Entender essa fronteira é metade do
que se cobra numa entrevista sobre feature stores.

POR QUE o ponto mais importante aqui é o **point-in-time correctness**: para cada
transação no instante `t`, calculamos o perfil do cliente usando APENAS as
transações ANTERIORES a `t` (estritamente). Se incluíssemos a transação atual (ou
futuras), o modelo aprenderia com informação que não existiria no momento real da
decisão — *data leakage* — e teria métricas ótimas no treino e péssimas em
produção. A linha `.shift(1)` e o `- 1` nas contagens são, literalmente, a
implementação dessa garantia.

A saída é uma SÉRIE TEMPORAL de perfis (uma linha por transação), não um snapshot
único por cliente. É exatamente isso que permite ao Feast fazer o join
"as-of-event-time" depois (ver `offline.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_fs.config import CUSTOMER_FEATURES, RAW_TRANSACTIONS


def _features_for_account(g: pd.DataFrame) -> pd.DataFrame:
    """Calcula, para uma conta, o perfil as-of-cada-transação (só com o passado)."""
    g = g.sort_values("event_timestamp").copy()
    amt = g["amount"]

    # Estatísticas históricas EXCLUINDO a transação atual:
    # expanding() acumula da 1ª linha até a atual; .shift(1) joga isso para a
    # linha seguinte, de modo que cada linha enxerga só o que veio antes dela.
    g["amount_avg"] = amt.expanding().mean().shift(1)
    g["amount_std"] = amt.expanding().std().shift(1)
    # Número de transações anteriores (0 na primeira, 1 na segunda, ...).
    g["txn_count_lifetime"] = np.arange(len(g), dtype="int64")

    # Contagens em janelas de tempo. rolling(tempo) inclui a linha atual no
    # extremo direito; subtraímos 1 para contar só as ANTERIORES dentro da janela.
    gi = g.set_index("event_timestamp")
    g["txn_count_1h"] = (gi["amount"].rolling("1h").count().to_numpy() - 1).astype("int64")
    g["txn_count_24h"] = (gi["amount"].rolling("24h").count().to_numpy() - 1).astype("int64")

    # Estado da transação ANTERIOR (para distância/velocidade calculadas online).
    g["last_txn_lat"] = g["lat"].shift(1)
    g["last_txn_lon"] = g["lon"].shift(1)
    prev_ts = g["event_timestamp"].shift(1)
    g["last_txn_unixtime"] = prev_ts.apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
    return g


def compute(write: bool = True) -> pd.DataFrame:
    """Lê as transações cruas e produz a tabela de features do perfil do cliente."""
    txns = pd.read_parquet(RAW_TRANSACTIONS)

    # Iteração explícita por conta (em vez de groupby.apply) para manter a coluna
    # de agrupamento disponível e evitar ambiguidades de versão do pandas.
    parts = [_features_for_account(g) for _, g in txns.groupby("account_id", sort=False)]
    feats = pd.concat(parts, ignore_index=True)

    out = feats[
        [
            "account_id",
            "event_timestamp",
            "amount_avg",
            "amount_std",
            "txn_count_1h",
            "txn_count_24h",
            "txn_count_lifetime",
            "last_txn_lat",
            "last_txn_lon",
            "last_txn_unixtime",
        ]
    ].copy()
    # created_timestamp exigido pelo FileSource do Feast.
    out["created"] = feats["event_timestamp"]

    if write:
        CUSTOMER_FEATURES.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(CUSTOMER_FEATURES, index=False)
    return out


def main() -> None:
    out = compute(write=True)
    print(f"Calculadas features de perfil para {len(out):,} transações.")
    print(f"Colunas: {[c for c in out.columns if c not in ('account_id', 'created')]}")
    print(f"Escrito em {CUSTOMER_FEATURES}")


if __name__ == "__main__":
    main()
