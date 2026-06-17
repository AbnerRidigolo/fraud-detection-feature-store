"""Testes da geração de dados: schema, reprodutibilidade e presença de fraude."""

import pandas as pd

from fraud_fs import generate_data

REQUIRED_COLUMNS = {
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
}


def _small(monkeypatch):
    """Reduz a escala para o teste rodar rápido, sem tocar em disco."""
    monkeypatch.setattr(generate_data, "N_CUSTOMERS", 15)
    monkeypatch.setattr(generate_data, "N_DAYS", 10)


def test_schema_and_ranges(monkeypatch):
    _small(monkeypatch)
    df = generate_data.generate(write=False)
    assert REQUIRED_COLUMNS.issubset(df.columns)
    assert (df["amount"] > 0).all()
    assert df["lat"].between(-90, 90).all()
    assert df["lon"].between(-180, 180).all()
    assert df["event_timestamp"].is_monotonic_increasing  # ordenado por tempo


def test_contains_both_classes(monkeypatch):
    _small(monkeypatch)
    df = generate_data.generate(write=False)
    assert df["is_fraud"].nunique() == 2  # tem legítimas E fraudes
    assert df["is_fraud"].sum() > 0


def test_deterministic_given_seed(monkeypatch):
    _small(monkeypatch)
    a = generate_data.generate(write=False)
    b = generate_data.generate(write=False)
    pd.testing.assert_frame_equal(a, b)
