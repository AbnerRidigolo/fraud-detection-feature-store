"""Testes das transformações puras.

Estas funções são compartilhadas entre os caminhos offline e online, então testá-las
isoladamente é o que nos dá confiança de que a feature significa a mesma coisa nos
dois lugares (anti training/serving skew).
"""

import numpy as np

from fraud_fs import transforms


def test_haversine_known_distance_sp_rio():
    # São Paulo -> Rio de Janeiro: ~360 km em linha reta.
    d = transforms.haversine_km(-23.55, -46.63, -22.91, -43.17)
    assert 340 < d < 380


def test_haversine_same_point_is_zero():
    assert transforms.haversine_km(10.0, 20.0, 10.0, 20.0) == 0.0


def test_haversine_one_degree_longitude_at_equator():
    # 1 grau de longitude no equador ~ 111.19 km.
    d = transforms.haversine_km(0.0, 0.0, 0.0, 1.0)
    assert abs(d - 111.19) < 1.0


def test_amount_zscore():
    # (100 - 50) / 10 = 5 (o eps é desprezível).
    assert abs(transforms.amount_zscore(100.0, 50.0, 10.0) - 5.0) < 1e-3


def test_velocity_100km_in_one_hour():
    assert abs(transforms.velocity_kmh(100.0, 3600.0) - 100.0) < 1e-2


def test_velocity_zero_seconds_is_finite_and_large():
    # Guarda contra divisão por zero: deve ser finito e altíssimo, não inf/NaN.
    v = transforms.velocity_kmh(100.0, 0.0)
    assert np.isfinite(v) and v > 1e6


def test_transforms_are_vectorized():
    import pandas as pd

    s = pd.Series([100.0, 200.0])
    out = transforms.amount_to_avg_ratio(s, pd.Series([50.0, 50.0]))
    assert list(np.round(out, 1)) == [2.0, 4.0]
