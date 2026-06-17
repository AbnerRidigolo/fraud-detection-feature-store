"""Inferência ONLINE: lê features do Redis em baixa latência e pontua a transação.

Aqui o ciclo se fecha. No treino usamos `get_historical_features` (offline,
point-in-time). Na inferência usamos `get_online_features` (online, último valor)
— mas pedindo EXATAMENTE as mesmas referências de feature (`FEATURE_REFS`) e
montando o vetor na MESMA ordem (`MODEL_FEATURES`). É essa simetria que garante
que o modelo veja em produção o mesmo espaço de features do treino.

Fluxo de uma decisão:
1. Chega uma transação: conta, valor, lat, lon, hora.
2. `get_online_features` busca no Redis o perfil já materializado da conta e, em
   cima dele + os dados do request, calcula as on-demand features (z-score,
   distância, velocidade) na hora.
3. O modelo carregado pontua o vetor e devolve probabilidade + decisão.
"""

from __future__ import annotations

from datetime import UTC, datetime

import joblib
import pandas as pd

from fraud_fs.config import FEATURE_REFS, MODEL_PATH
from fraud_fs.store import get_store

_BUNDLE = None


def _load_bundle() -> dict:
    """Carrega (uma vez) o modelo, a lista de features e o threshold."""
    global _BUNDLE
    if _BUNDLE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {MODEL_PATH}. Rode o treino antes "
                "(`python -m fraud_fs.train`)."
            )
        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE


def score_transaction(
    account_id: str,
    amount: float,
    lat: float,
    lon: float,
    timestamp: datetime | None = None,
) -> dict:
    """Pontua uma transação consultando a online store + on-demand transforms."""
    bundle = _load_bundle()
    store = get_store()

    ts = timestamp or datetime.now(UTC)
    entity_row = {
        "account_id": account_id,
        "amount": float(amount),
        "lat": float(lat),
        "lon": float(lon),
        "txn_unixtime": ts.timestamp(),
    }

    feats = store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=[entity_row],
    ).to_dict()

    # Monta o vetor na ordem exata em que o modelo foi treinado.
    row = {name: feats[name][0] for name in bundle["features"]}
    x = pd.DataFrame([row])[bundle["features"]]

    proba = float(bundle["model"].predict_proba(x)[:, 1][0])
    return {
        "account_id": account_id,
        "fraud_probability": proba,
        "is_fraud": bool(proba >= bundle["threshold"]),
        "threshold": bundle["threshold"],
        "features": row,
    }


def _demo() -> None:
    """Demonstra o serving online: uma transação normal vs. uma suspeita."""
    from datetime import timedelta

    from fraud_fs.config import RAW_TRANSACTIONS

    txns = pd.read_parquet(RAW_TRANSACTIONS)
    # Conta com mais histórico (perfil bem materializado no Redis).
    acc = txns["account_id"].value_counts().idxmax()
    hist = txns[txns["account_id"] == acc].sort_values("event_timestamp")
    last = hist.iloc[-1]
    home_lat, home_lon = float(last["lat"]), float(last["lon"])
    avg = float(hist["amount"].mean())
    # Logo após a última transação conhecida — assim a velocidade faz sentido.
    soon = last["event_timestamp"].to_pydatetime() + timedelta(minutes=10)

    print(f"=== Demo de inferência online (conta {acc}) ===")
    normal = score_transaction(acc, amount=avg, lat=home_lat, lon=home_lon, timestamp=soon)
    print(
        f"[normal]   valor=R${avg:7.2f} perto de casa     "
        f"p(fraude)={normal['fraud_probability']:.3f} -> fraude={normal['is_fraud']}"
    )

    # Valor altíssimo + Tóquio 10 min depois de uma compra local => impossível.
    suspeita = score_transaction(acc, amount=avg * 80, lat=35.68, lon=139.69, timestamp=soon)
    print(
        f"[suspeita] valor=R${avg * 80:7.2f} em Tóquio 10min depois "
        f"p(fraude)={suspeita['fraud_probability']:.3f} -> fraude={suspeita['is_fraud']}"
    )
    f = suspeita["features"]
    print(f"  z-score do valor : {f.get('amount_zscore'):.1f}")
    print(f"  distância (km)   : {f.get('distance_from_last_km'):.0f}")
    print(f"  velocidade (km/h): {f.get('velocity_kmh'):,.0f}  <- viagem impossível")


if __name__ == "__main__":
    _demo()
