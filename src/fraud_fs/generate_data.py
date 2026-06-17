"""Geração de transações sintéticas com padrões de fraude injetados.

POR QUE dados sintéticos (e não um CSV do Kaggle): para um projeto de portfólio
sobre *feature store*, o valor está em controlar a VERDADE DE BASE. Como nós
mesmos injetamos a fraude, sabemos exatamente qual sinal cada feature deveria
capturar — dá pra provar que a feature store funciona, não só torcer.

Modelamos cada cliente com um PERFIL próprio (cidade-sede, ticket médio,
volatilidade de gasto). Fraude não é um valor alto em termos absolutos; é um
desvio do *normal daquele cliente*. Por isso injetamos três padrões clássicos:

1. **Valor fora do perfil** — uma compra muito acima do ticket médio do cliente.
2. **Sequência rápida** (card testing) — uma rajada de transações em minutos.
3. **Localização impossível** — uma compra em outro continente minutos depois
   de uma local, implicando uma velocidade de deslocamento impossível.

Tudo é semeado (`SEED`) para ser 100% reprodutível: rodar de novo dá o mesmo
dataset, o que é essencial para testes determinísticos e para um portfólio que
qualquer um consegue reproduzir.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from faker import Faker

from fraud_fs.config import (
    AVG_TXN_PER_DAY,
    FRAUD_RATE,
    N_CUSTOMERS,
    N_DAYS,
    RAW_TRANSACTIONS,
    SEED,
)

# Cidades-sede plausíveis (lat, lon) — a maioria no Brasil para dar coesão ao
# dataset; transações normais acontecem perto da sede do cliente.
HOME_CITIES = [
    ("Sao Paulo", -23.55, -46.63),
    ("Rio de Janeiro", -22.91, -43.17),
    ("Belo Horizonte", -19.92, -43.94),
    ("Curitiba", -25.43, -49.27),
    ("Porto Alegre", -30.03, -51.23),
    ("Salvador", -12.97, -38.50),
    ("Recife", -8.05, -34.88),
    ("Brasilia", -15.79, -47.88),
    ("Fortaleza", -3.73, -38.52),
    ("Manaus", -3.12, -60.02),
]

# Cidades distantes usadas no padrão "localização impossível".
FAR_CITIES = [
    ("Tokyo", 35.68, 139.69),
    ("London", 51.51, -0.13),
    ("New York", 40.71, -74.01),
    ("Sydney", -33.87, 151.21),
    ("Dubai", 25.20, 55.27),
]

MERCHANT_CATEGORIES = [
    "grocery",
    "restaurant",
    "fuel",
    "pharmacy",
    "electronics",
    "clothing",
    "travel",
    "entertainment",
    "utilities",
    "transport",
]


def _make_customers(rng: np.random.Generator) -> pd.DataFrame:
    """Cria N_CUSTOMERS clientes, cada um com sede e perfil de gasto próprios."""
    rows = []
    for i in range(N_CUSTOMERS):
        city, lat, lon = HOME_CITIES[rng.integers(len(HOME_CITIES))]
        # Ticket médio log-normal: maioria gasta pouco, alguns gastam muito.
        mean_amount = float(np.exp(rng.normal(4.5, 0.6)))  # ~R$90, cauda longa
        rows.append(
            {
                "account_id": f"ACC{i:06d}",
                "home_city": city,
                "home_lat": lat,
                "home_lon": lon,
                "mean_amount": mean_amount,
                "std_amount": mean_amount * rng.uniform(0.2, 0.5),
            }
        )
    return pd.DataFrame(rows)


def _legit_transactions(
    customers: pd.DataFrame, rng: np.random.Generator, fake: Faker, start: pd.Timestamp
) -> pd.DataFrame:
    """Gera o tráfego LEGÍTIMO: compras perto de casa, dentro do perfil."""
    rows = []
    horizon_seconds = N_DAYS * 24 * 3600
    for c in customers.itertuples():
        n = rng.poisson(N_DAYS * AVG_TXN_PER_DAY)
        for _ in range(int(n)):
            offset = int(rng.integers(0, horizon_seconds))
            ts = start + pd.Timedelta(seconds=offset)
            amount = max(1.0, float(rng.normal(c.mean_amount, c.std_amount)))
            # Pequeno jitter geográfico em torno da sede (~ dezenas de km).
            lat = c.home_lat + rng.normal(0, 0.05)
            lon = c.home_lon + rng.normal(0, 0.05)
            rows.append(
                {
                    "account_id": c.account_id,
                    "event_timestamp": ts,
                    "amount": round(amount, 2),
                    "merchant": fake.company(),
                    "merchant_category": MERCHANT_CATEGORIES[
                        rng.integers(len(MERCHANT_CATEGORIES))
                    ],
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "is_fraud": 0,
                }
            )
    return pd.DataFrame(rows)


def _inject_amount_fraud(
    df: pd.DataFrame, idx: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Padrão 1: marca transações existentes como valores muito acima do perfil."""
    df = df.copy()
    df.loc[idx, "amount"] = (df.loc[idx, "amount"] * rng.uniform(8, 40, size=len(idx))).round(2)
    df.loc[idx, "is_fraud"] = 1
    return df


def _make_burst_fraud(
    customers: pd.DataFrame, rng: np.random.Generator, fake: Faker, start: pd.Timestamp, n: int
) -> list[dict]:
    """Padrão 2: rajadas de muitas transações pequenas em poucos minutos."""
    rows: list[dict] = []
    for _ in range(n):
        c = customers.iloc[rng.integers(len(customers))]
        base = start + pd.Timedelta(seconds=int(rng.integers(0, N_DAYS * 24 * 3600)))
        for k in range(int(rng.integers(5, 12))):
            ts = base + pd.Timedelta(seconds=int(k * rng.integers(10, 40)))
            rows.append(
                {
                    "account_id": c.account_id,
                    "event_timestamp": ts,
                    "amount": round(float(rng.uniform(1, 20)), 2),  # card testing = valores baixos
                    "merchant": fake.company(),
                    "merchant_category": "electronics",
                    "lat": round(c.home_lat + rng.normal(0, 0.05), 5),
                    "lon": round(c.home_lon + rng.normal(0, 0.05), 5),
                    "is_fraud": 1,
                }
            )
    return rows


def _make_geo_fraud(
    customers: pd.DataFrame, rng: np.random.Generator, fake: Faker, start: pd.Timestamp, n: int
) -> list[dict]:
    """Padrão 3: compra em cidade distante minutos depois de uma local."""
    rows: list[dict] = []
    for _ in range(n):
        c = customers.iloc[rng.integers(len(customers))]
        base = start + pd.Timedelta(seconds=int(rng.integers(0, N_DAYS * 24 * 3600)))
        # Transação local "âncora".
        rows.append(
            {
                "account_id": c.account_id,
                "event_timestamp": base,
                "amount": round(float(rng.normal(c.mean_amount, c.std_amount)), 2),
                "merchant": fake.company(),
                "merchant_category": "restaurant",
                "lat": round(c.home_lat + rng.normal(0, 0.05), 5),
                "lon": round(c.home_lon + rng.normal(0, 0.05), 5),
                "is_fraud": 0,
            }
        )
        # Minutos depois, do outro lado do mundo => viagem impossível.
        _, flat, flon = FAR_CITIES[rng.integers(len(FAR_CITIES))]
        rows.append(
            {
                "account_id": c.account_id,
                "event_timestamp": base + pd.Timedelta(minutes=int(rng.integers(2, 30))),
                "amount": round(float(rng.uniform(50, 800)), 2),
                "merchant": fake.company(),
                "merchant_category": "electronics",
                "lat": round(flat + rng.normal(0, 0.05), 5),
                "lon": round(flon + rng.normal(0, 0.05), 5),
                "is_fraud": 1,
            }
        )
    return rows


def generate(write: bool = True) -> pd.DataFrame:
    """Gera o dataset completo e (opcionalmente) grava em parquet."""
    rng = np.random.default_rng(SEED)
    fake = Faker("pt_BR")
    Faker.seed(SEED)

    start = pd.Timestamp("2025-01-01", tz="UTC")
    customers = _make_customers(rng)
    legit = _legit_transactions(customers, rng, fake, start)

    # Orçamento total de fraude, repartido entre os três padrões.
    n_fraud_total = max(3, int(len(legit) * FRAUD_RATE))
    n_amount = n_fraud_total // 2
    n_burst = max(1, n_fraud_total // 8)  # cada rajada gera ~5-12 linhas
    n_geo = max(1, n_fraud_total // 6)  # cada caso gera 2 linhas

    amount_idx = rng.choice(legit.index.values, size=min(n_amount, len(legit)), replace=False)
    legit = _inject_amount_fraud(legit, amount_idx, rng)

    burst = _make_burst_fraud(customers, rng, fake, start, n_burst)
    geo = _make_geo_fraud(customers, rng, fake, start, n_geo)

    df = pd.concat([legit, pd.DataFrame(burst + geo)], ignore_index=True)
    df = df.sort_values("event_timestamp").reset_index(drop=True)

    # IDs determinísticos derivados do índice (UUID5 = estável entre execuções).
    df["transaction_id"] = [str(uuid.uuid5(uuid.NAMESPACE_OID, f"txn-{i}")) for i in range(len(df))]
    # Coluna de ingestão exigida pelo FileSource do Feast (created_timestamp).
    df["created"] = df["event_timestamp"]

    df = df[
        [
            "transaction_id",
            "account_id",
            "event_timestamp",
            "amount",
            "merchant",
            "merchant_category",
            "lat",
            "lon",
            "is_fraud",
            "created",
        ]
    ]

    if write:
        RAW_TRANSACTIONS.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RAW_TRANSACTIONS, index=False)
    return df


def main() -> None:
    df = generate(write=True)
    fraud = int(df["is_fraud"].sum())
    print(f"Geradas {len(df):,} transações de {df['account_id'].nunique()} contas.")
    print(f"Fraudes: {fraud:,} ({fraud / len(df):.2%}) — escrito em {RAW_TRANSACTIONS}")


if __name__ == "__main__":
    main()
