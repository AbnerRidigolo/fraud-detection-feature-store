"""Operações de ciclo de vida do Feast: `apply` e `materialize`.

POR QUE estes dois verbos são o coração operacional de uma feature store:

* **apply** — lê as definições (`feature_repo/definitions.py`) e as registra no
  *registry*. É o "deploy" do contrato de features: nada existe para o Feast até
  ser aplicado. Idempotente — reaplicar só reconcilia o que mudou.

* **materialize** — copia os valores mais recentes de cada feature da OFFLINE
  store (Parquet histórico) para a ONLINE store (Redis). Este é o passo que
  muita gente não entende: o online store não se popula sozinho. Materializar é
  o que transforma "tenho o histórico em disco" em "consigo ler o último valor
  da conta X em 1 ms". Em produção isso roda agendado (cron/airflow) ou em
  streaming; aqui rodamos uma vez sobre toda a janela.
"""

from __future__ import annotations

import sys
from datetime import timedelta

import pandas as pd

from fraud_fs.config import FEATURE_REPO_DIR, RAW_TRANSACTIONS
from fraud_fs.store import get_store


def _load_definitions():
    """Importa o módulo de definições do feature_repo (não é um pacote no path)."""
    sys.path.insert(0, str(FEATURE_REPO_DIR))
    import definitions as d  # type: ignore

    return d


def apply_repo() -> None:
    """Registra entidades e feature views no registry (equivale a `feast apply`)."""
    store = get_store()
    d = _load_definitions()
    store.apply([d.account, d.customer_profile, d.transaction_request, d.transaction_risk])
    print("Feast apply concluído: entidade + feature views registradas no registry.")


def materialize_online() -> None:
    """Empurra os valores mais recentes da offline store para o Redis (online)."""
    store = get_store()
    txns = pd.read_parquet(RAW_TRANSACTIONS, columns=["event_timestamp"])
    start = txns["event_timestamp"].min().to_pydatetime()
    # +1s para garantir que o último evento entra na janela [start, end].
    end = txns["event_timestamp"].max().to_pydatetime() + timedelta(seconds=1)
    store.materialize(start_date=start, end_date=end)
    print(f"Materialização concluída: perfis de {start.date()} a {end.date()} no Redis.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Operações do Feast")
    parser.add_argument("step", choices=["apply", "materialize", "all"])
    args = parser.parse_args()

    if args.step in ("apply", "all"):
        apply_repo()
    if args.step in ("materialize", "all"):
        materialize_online()


if __name__ == "__main__":
    main()
