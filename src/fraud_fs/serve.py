"""API de serving online (FastAPI) — expõe a pontuação de fraude como serviço.

POR QUE uma API e não só um script: serving online de verdade é um serviço de
baixa latência que outros sistemas (gateway de pagamento, app) chamam de forma
síncrona durante a transação. Esta API demonstra esse papel: recebe a transação,
consulta a online store (Redis), calcula as on-demand features e devolve a
decisão em um único POST.

Suba com:  uvicorn fraud_fs.serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_fs import inference

app = FastAPI(
    title="Fraud Feature Store — Online Scoring",
    description="Pontuação de fraude em tempo real usando Feast (Redis) + scikit-learn.",
    version="0.1.0",
)


class Transaction(BaseModel):
    """Contrato de entrada — Pydantic valida tipos/limites na fronteira."""

    account_id: str = Field(..., examples=["ACC000001"])
    amount: float = Field(..., gt=0, examples=[129.90])
    lat: float = Field(..., ge=-90, le=90, examples=[-23.55])
    lon: float = Field(..., ge=-180, le=180, examples=[-46.63])
    timestamp: datetime | None = Field(default=None, description="UTC; default = agora.")


class ScoreResponse(BaseModel):
    account_id: str
    fraud_probability: float
    is_fraud: bool
    threshold: float


@app.get("/health")
def health() -> dict:
    """Liveness simples para Docker/orquestrador."""
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    """Pontua uma transação consultando a online store + modelo."""
    try:
        result = inference.score_transaction(
            account_id=txn.account_id,
            amount=txn.amount,
            lat=txn.lat,
            lon=txn.lon,
            timestamp=txn.timestamp or datetime.now(UTC),
        )
    except FileNotFoundError as exc:
        # Modelo ainda não treinado: 503 (serviço indisponível), não 500.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ScoreResponse(**{k: result[k] for k in ScoreResponse.model_fields})
