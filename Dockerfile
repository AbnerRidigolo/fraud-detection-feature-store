# Imagem única usada tanto pelo pipeline one-shot quanto pela API de serving.
# (Mesma imagem, comandos diferentes no docker-compose — menos superfície de drift.)
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REDIS_CONNECTION_STRING=redis:6379

# Dependências primeiro (camada cacheável enquanto o requirements não muda).
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Código da aplicação.
COPY pyproject.toml README.md ./
COPY src ./src
COPY feature_repo ./feature_repo
COPY scripts ./scripts
RUN pip install -e . && chmod +x scripts/*.sh

# Default: roda o ciclo completo. O compose sobrescreve para a API.
CMD ["bash", "scripts/run_pipeline.sh"]
