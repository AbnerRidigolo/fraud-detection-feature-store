#!/usr/bin/env bash
# Orquestração one-shot do ciclo completo da feature store.
# Cada passo é um módulo independente — dá pra rodar isolado também (ver Makefile).
set -euo pipefail

echo "==> 1/6  Gerando transações sintéticas (Faker + fraude injetada)"
python -m fraud_fs.generate_data

echo "==> 2/6  Calculando features de perfil (offline, point-in-time)"
python -m fraud_fs.compute_features

echo "==> 3/6  feast apply — registrando definições no registry"
python -m fraud_fs.pipeline apply

echo "==> 4/6  Materializando perfis para o online store (Redis)"
python -m fraud_fs.pipeline materialize

echo "==> 5/6  Montando dataset de treino (point-in-time join) e treinando"
python -m fraud_fs.offline
python -m fraud_fs.train

echo "==> 6/6  Inferência online (demo: transação normal vs. suspeita)"
python -m fraud_fs.inference

echo "==> Pipeline concluído."
