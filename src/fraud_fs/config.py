"""Configuração central: caminhos, escala dos dados e referências de feature.

Centralizar isto resolve dois problemas reais:

* **Fonte única de verdade para nomes de feature.** As mesmas features são
  pedidas no treino (offline) e na inferência (online). Se a lista divergir
  entre os dois lugares, o vetor que entra no modelo muda silenciosamente. Aqui
  ela é declarada UMA vez (`FEATURE_REFS`) e derivada para os dois caminhos.
* **Parametrização por ambiente.** Em CI/demo a gente quer rodar pequeno e
  rápido; localmente, maior. Tudo é lido de variáveis de ambiente com defaults
  sãos, então nada precisa ser editado no código para mudar a escala.
"""

from __future__ import annotations

import os
from pathlib import Path

# fraud-feature-store/ (sobe de src/fraud_fs/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("FRAUD_FS_DATA_DIR", str(PROJECT_ROOT / "data")))
FEATURE_REPO_DIR = PROJECT_ROOT / "feature_repo"

# Artefatos do pipeline (todos gerados; nada disso é versionado no git).
RAW_TRANSACTIONS = DATA_DIR / "transactions.parquet"
CUSTOMER_FEATURES = DATA_DIR / "customer_features.parquet"
TRAINING_DATASET = DATA_DIR / "training_dataset.parquet"
MODEL_PATH = DATA_DIR / "model.joblib"

# --- Escala da geração de dados (overridável por env para CI/demo) -----------
N_CUSTOMERS = int(os.environ.get("FRAUD_FS_N_CUSTOMERS", "300"))
N_DAYS = int(os.environ.get("FRAUD_FS_N_DAYS", "60"))
AVG_TXN_PER_DAY = float(os.environ.get("FRAUD_FS_TXN_RATE", "1.2"))
FRAUD_RATE = float(os.environ.get("FRAUD_FS_FRAUD_RATE", "0.015"))
SEED = int(os.environ.get("FRAUD_FS_SEED", "42"))

# --- Infra -------------------------------------------------------------------
# Lido aqui para que `store.get_store()` consiga repassar ao feature_store.yaml
# via interpolação de ${REDIS_CONNECTION_STRING}.
REDIS_CONNECTION_STRING = os.environ.get("REDIS_CONNECTION_STRING", "localhost:6379")

# --- Referências de feature (FONTE ÚNICA DE VERDADE) -------------------------
# Features pré-computadas em lote, vindas da FeatureView `customer_profile`.
PROFILE_FEATURE_REFS = [
    "customer_profile:amount_avg",
    "customer_profile:amount_std",
    "customer_profile:txn_count_1h",
    "customer_profile:txn_count_24h",
    "customer_profile:txn_count_lifetime",
    "customer_profile:last_txn_lat",
    "customer_profile:last_txn_lon",
    "customer_profile:last_txn_unixtime",
]
# Features calculadas em tempo de request pela on-demand feature view.
RISK_FEATURE_REFS = [
    "transaction_risk:amount_zscore",
    "transaction_risk:amount_to_avg_ratio",
    "transaction_risk:distance_from_last_km",
    "transaction_risk:seconds_since_last",
    "transaction_risk:velocity_kmh",
]
FEATURE_REFS = PROFILE_FEATURE_REFS + RISK_FEATURE_REFS

# Nomes "nus" das colunas, na ordem exata em que o modelo as consome.
# Derivado de FEATURE_REFS para que treino e inferência nunca saiam de sincronia.
MODEL_FEATURES = [ref.split(":", 1)[1] for ref in FEATURE_REFS]
