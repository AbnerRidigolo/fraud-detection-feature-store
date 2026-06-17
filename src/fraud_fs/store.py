"""Fábrica do objeto `FeatureStore` do Feast.

Por que um helper em vez de instanciar `FeatureStore` em cada módulo:

* O `feature_store.yaml` referencia `${REDIS_CONNECTION_STRING}` e
  `${FRAUD_FS_REGISTRY}`. Essas variáveis precisam existir no ambiente ANTES de
  o Feast ler o YAML. Concentrar isso num único ponto evita o erro sutil de
  "esqueci de exportar a env e o Feast tentou conectar no host errado".
* O mesmo código funciona local (`localhost:6379`), em Docker (`redis:6379`) e
  em CI (serviço `redis`) trocando só a env — sem if/else espalhado.
"""

from __future__ import annotations

import os

from feast import FeatureStore

from fraud_fs.config import DATA_DIR, FEATURE_REPO_DIR, REDIS_CONNECTION_STRING


def get_store() -> FeatureStore:
    """Devolve um FeatureStore configurado a partir do ambiente."""
    os.environ.setdefault("REDIS_CONNECTION_STRING", REDIS_CONNECTION_STRING)
    os.environ.setdefault("FRAUD_FS_REGISTRY", str(DATA_DIR / "registry.db"))
    return FeatureStore(repo_path=str(FEATURE_REPO_DIR))
