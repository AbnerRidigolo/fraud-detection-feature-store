"""Definições do Feast: entidade, fontes, feature views (batch + on-demand).

Este arquivo é o "contrato" das features. O `feast apply` lê estas definições e
as registra no *registry*. Conceitos-chave que aparecem aqui:

* **Entity** — o "sobre o quê" das features. Aqui a entidade é a CONTA
  (`account_id`); toda feature é uma propriedade de uma conta num instante.
* **FileSource** — aponta para os dados físicos (Parquet) na offline store e diz
  ao Feast qual coluna é o *event_timestamp* (a hora do evento, usada no
  point-in-time join) e qual é o *created* (hora de ingestão).
* **FeatureView** (`customer_profile`) — agrupa features que vêm da MESMA fonte e
  compartilham a mesma entidade e TTL. É a unidade que se materializa para o
  online store e que se pede no treino.
* **RequestSource** — dados que só existem no instante do request (o valor, a
  geolocalização e a hora DESTA transação). Não dá para pré-computar: dependem da
  transação que está chegando agora.
* **OnDemandFeatureView** (`transaction_risk`) — features calculadas em tempo de
  request combinando o RequestSource com o perfil já armazenado. É aqui que mora
  o z-score do valor, a distância e a velocidade ("viagem impossível"). Usa
  EXATAMENTE as mesmas funções de `fraud_fs.transforms` que o cálculo offline —
  é isso que elimina training/serving skew.
"""

from __future__ import annotations

import pathlib
import sys
from datetime import timedelta

import pandas as pd

# Garante que `fraud_fs` é importável quando o Feast carrega este arquivo por path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from feast import Entity, FeatureView, Field, FileSource, RequestSource, ValueType  # noqa: E402
from feast.on_demand_feature_view import on_demand_feature_view  # noqa: E402
from feast.types import Float64, Int64  # noqa: E402

from fraud_fs import transforms  # noqa: E402
from fraud_fs.config import CUSTOMER_FEATURES  # noqa: E402

# --- Entidade ----------------------------------------------------------------
account = Entity(
    name="account",
    join_keys=["account_id"],
    value_type=ValueType.STRING,
    description="Conta/cliente — a chave sobre a qual todas as features são agregadas.",
)

# --- Fonte offline (a tabela de perfis calculada por compute_features.py) -----
customer_features_source = FileSource(
    name="customer_features_source",
    path=str(CUSTOMER_FEATURES),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# --- Feature view em lote: o perfil histórico do cliente ---------------------
customer_profile = FeatureView(
    name="customer_profile",
    entities=[account],
    # TTL: por quanto tempo um valor de feature continua "válido". No
    # point-in-time join, o Feast só casa um perfil cujo event_timestamp esteja
    # dentro de `ttl` antes do evento — evita usar um perfil obsoleto demais.
    ttl=timedelta(days=90),
    schema=[
        Field(name="amount_avg", dtype=Float64),
        Field(name="amount_std", dtype=Float64),
        Field(name="txn_count_1h", dtype=Int64),
        Field(name="txn_count_24h", dtype=Int64),
        Field(name="txn_count_lifetime", dtype=Int64),
        Field(name="last_txn_lat", dtype=Float64),
        Field(name="last_txn_lon", dtype=Float64),
        Field(name="last_txn_unixtime", dtype=Float64),
    ],
    online=True,  # materializável para o Redis
    source=customer_features_source,
)

# --- Dados disponíveis só no request (a transação que está chegando) ---------
transaction_request = RequestSource(
    name="transaction_request",
    schema=[
        Field(name="amount", dtype=Float64),
        Field(name="lat", dtype=Float64),
        Field(name="lon", dtype=Float64),
        Field(name="txn_unixtime", dtype=Float64),
    ],
)


# --- On-demand feature view: features de risco em tempo de request -----------
@on_demand_feature_view(
    sources=[customer_profile, transaction_request],
    schema=[
        Field(name="amount_zscore", dtype=Float64),
        Field(name="amount_to_avg_ratio", dtype=Float64),
        Field(name="distance_from_last_km", dtype=Float64),
        Field(name="seconds_since_last", dtype=Float64),
        Field(name="velocity_kmh", dtype=Float64),
    ],
)
def transaction_risk(inputs: pd.DataFrame) -> pd.DataFrame:
    """Combina o perfil armazenado com a transação atual para medir o risco.

    Recebe um DataFrame cujas colunas são as features do `customer_profile` MAIS
    os campos do `transaction_request`. As mesmas fórmulas de `transforms` rodam
    aqui (online) e no cálculo offline — garantia de consistência.
    """
    out = pd.DataFrame()
    out["amount_zscore"] = transforms.amount_zscore(
        inputs["amount"], inputs["amount_avg"], inputs["amount_std"]
    )
    out["amount_to_avg_ratio"] = transforms.amount_to_avg_ratio(
        inputs["amount"], inputs["amount_avg"]
    )
    out["distance_from_last_km"] = transforms.haversine_km(
        inputs["lat"], inputs["lon"], inputs["last_txn_lat"], inputs["last_txn_lon"]
    )
    out["seconds_since_last"] = inputs["txn_unixtime"] - inputs["last_txn_unixtime"]
    out["velocity_kmh"] = transforms.velocity_kmh(
        out["distance_from_last_km"], out["seconds_since_last"]
    )
    return out
