"""Testes do cálculo de features — com foco em PROVAR que não há leakage.

O teste mais importante deste projeto: garantir que a feature de uma transação no
instante `t` foi calculada usando APENAS transações anteriores a `t`. Se este
teste passar, a base do point-in-time correctness está correta.
"""

import numpy as np
import pandas as pd

from fraud_fs.compute_features import _features_for_account


def _toy_account() -> pd.DataFrame:
    """Quatro transações da mesma conta, em ordem cronológica conhecida."""
    base = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "account_id": ["ACC1"] * 4,
            "event_timestamp": [
                base,
                base + pd.Timedelta(minutes=10),  # +10min (dentro de 1h)
                base + pd.Timedelta(minutes=30),  # +30min (dentro de 1h)
                base + pd.Timedelta(hours=2),  # +2h (fora de 1h)
            ],
            "amount": [100.0, 200.0, 300.0, 400.0],
            "lat": [0.0, 0.0, 0.0, 10.0],
            "lon": [0.0, 0.0, 0.0, 10.0],
        }
    )


def test_first_transaction_has_empty_profile():
    g = _features_for_account(_toy_account())
    first = g.iloc[0]
    # Sem passado: média/desvio/última-transação são NaN; contagens são 0.
    assert np.isnan(first["amount_avg"])
    assert np.isnan(first["amount_std"])
    assert np.isnan(first["last_txn_lat"])
    assert first["txn_count_lifetime"] == 0
    assert first["txn_count_1h"] == 0


def test_no_leakage_average_uses_only_past():
    g = _features_for_account(_toy_account())
    # amount_avg na linha i deve ser a média de amounts[0:i] (estritamente antes).
    assert g.iloc[1]["amount_avg"] == 100.0  # média de [100]
    assert g.iloc[2]["amount_avg"] == 150.0  # média de [100, 200]
    assert g.iloc[3]["amount_avg"] == 200.0  # média de [100, 200, 300]


def test_count_lifetime_is_number_of_prior_txns():
    g = _features_for_account(_toy_account())
    assert list(g["txn_count_lifetime"]) == [0, 1, 2, 3]


def test_count_1h_window_excludes_current_and_old():
    g = _features_for_account(_toy_account())
    # t0: 0 anteriores | t1(+10m): 1 | t2(+30m): 2 | t3(+2h): só t3 na janela => 0
    assert list(g["txn_count_1h"]) == [0, 1, 2, 0]


def test_last_txn_state_points_to_previous_row():
    g = _features_for_account(_toy_account())
    # A última transação registrada na linha i é a da linha i-1.
    assert g.iloc[3]["last_txn_lat"] == 0.0  # linha 2 tinha lat 0
    assert g.iloc[1]["last_txn_lat"] == 0.0
    # E o tempo da anterior bate com o timestamp da linha i-1.
    prev_unix = g.iloc[2]["event_timestamp"].timestamp()
    assert g.iloc[3]["last_txn_unixtime"] == prev_unix
