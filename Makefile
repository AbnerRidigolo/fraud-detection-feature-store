# Atalhos para rodar cada etapa isolada ou o pipeline inteiro.
# Uso típico local:  make setup && make pipeline
# Uso com Docker:    make up
PY ?= python

.PHONY: setup data features apply materialize offline train demo pipeline serve test lint fmt up down clean

setup:            ## Instala o pacote + ferramentas de dev (modo editável)
	pip install -e ".[dev]"

data:             ## 1. Gera transações sintéticas com fraude injetada
	$(PY) -m fraud_fs.generate_data

features:         ## 2. Calcula features de perfil (offline, point-in-time)
	$(PY) -m fraud_fs.compute_features

apply:            ## 3. Registra definições no registry (feast apply)
	$(PY) -m fraud_fs.pipeline apply

materialize:      ## 4. Materializa perfis para o online store (Redis)
	$(PY) -m fraud_fs.pipeline materialize

offline:          ## 5a. Monta dataset de treino (point-in-time join)
	$(PY) -m fraud_fs.offline

train:            ## 5b. Treina o modelo e avalia em hold-out temporal
	$(PY) -m fraud_fs.train

demo:             ## 6. Inferência online (normal vs. suspeita)
	$(PY) -m fraud_fs.inference

pipeline: data features apply materialize offline train demo  ## Ciclo completo

serve:            ## Sobe a API de scoring online
	uvicorn fraud_fs.serve:app --reload --port 8000

test:             ## Testes unitários (não precisam de Redis)
	$(PY) -m pytest -q

lint:             ## ruff + black (modo verificação)
	ruff check src tests feature_repo && black --check src tests feature_repo

fmt:              ## Aplica correções de ruff + black
	ruff check --fix src tests feature_repo && black src tests feature_repo

up:               ## Sobe tudo via Docker Compose (redis + pipeline + api)
	docker compose up --build

down:             ## Derruba os containers e volumes
	docker compose down -v

clean:            ## Remove artefatos gerados
	rm -f data/*.parquet data/*.db data/*.joblib
