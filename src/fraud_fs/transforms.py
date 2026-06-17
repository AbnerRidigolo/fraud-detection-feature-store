"""Transformações de feature puras, determinísticas e vetorizadas.

POR QUE este módulo existe isolado de tudo:

1. **Anti training/serving skew.** Estas funções são chamadas em DOIS lugares:
   no cálculo OFFLINE em lote (`compute_features.py`) e no cálculo ONLINE em
   tempo de request (a *on-demand feature view* em `feature_repo/definitions.py`).
   Se o treino e a inferência usarem fórmulas diferentes para a mesma feature, o
   modelo vê em produção números diferentes dos que viu no treino — esse é o bug
   mais clássico e mais caro de sistemas de ML. Centralizando a matemática num só
   lugar, garantimos por construção que offline e online são idênticos.

2. **Testabilidade.** Entram números, saem números. Sem I/O, sem Feast, sem
   pandas obrigatório. Isso torna cada feature trivial de cobrir com teste
   unitário (ver `tests/test_transforms.py`).

Todas as funções aceitam tanto escalares quanto `pandas.Series`/`numpy.ndarray`,
porque o caminho em lote opera sobre colunas inteiras e o caminho online também
recebe DataFrames (de uma ou mais linhas).
"""

from __future__ import annotations

import numpy as np

# Raio médio da Terra em km — usado na fórmula de Haversine.
EARTH_RADIUS_KM = 6371.0

# Epsilon para evitar divisão por zero sem mascarar o sinal do numerador.
_EPS = 1e-6


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em km entre dois pontos (lat/lon em graus) sobre a esfera.

    Usada para detectar "viagem impossível": se a distância entre a transação
    atual e a anterior é grande e o tempo entre elas é pequeno, a velocidade
    implícita fica fisicamente impossível (ver `velocity_kmh`).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def amount_zscore(amount, mean, std, eps: float = _EPS):
    """Quantos desvios-padrão o valor da transação está do perfil do cliente.

    Esta é a tradução direta de "desvio do padrão histórico": um z-score alto
    significa um valor muito atípico para AQUELE cliente (não em termos
    absolutos). R$ 5.000 é normal para uns, gritante para outros.
    """
    return (amount - mean) / (std + eps)


def amount_to_avg_ratio(amount, mean, eps: float = _EPS):
    """Razão entre o valor atual e o ticket médio histórico do cliente."""
    return amount / (mean + eps)


def velocity_kmh(distance_km, seconds, eps: float = _EPS):
    """Velocidade implícita (km/h) necessária para ir da transação anterior à atual.

    Valores absurdamente altos (ex.: > 1000 km/h, mais rápido que um avião)
    indicam que o cartão foi usado em dois lugares fisicamente incompatíveis no
    tempo decorrido — sinal forte de clonagem/fraude.
    """
    hours = np.maximum(seconds, 0.0) / 3600.0
    return distance_km / (hours + eps)
