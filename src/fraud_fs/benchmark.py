"""Mede a latência HTTP de POST /score com uma transação fixa e uma conexão."""

from __future__ import annotations

import argparse
import json
import os
import platform
from http.client import HTTPConnection, HTTPException
from math import ceil
from time import perf_counter_ns


def measure(
    connection: HTTPConnection, payload: bytes, requests: int, warmup: int
) -> dict[str, float]:
    headers = {"Content-Type": "application/json"}

    def post() -> None:
        connection.request("POST", "/score", body=payload, headers=headers)
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body.decode(errors='replace')}")

    for _ in range(warmup):
        post()

    samples_ms = []
    for _ in range(requests):
        start = perf_counter_ns()
        post()
        samples_ms.append((perf_counter_ns() - start) / 1_000_000)

    samples_ms.sort()
    # Percentis empíricos por nearest rank (posição começa em 1).
    return {
        "p50_ms": samples_ms[ceil(0.50 * requests) - 1],
        "p95_ms": samples_ms[ceil(0.95 * requests) - 1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--account-id", default="ACC000001")
    parser.add_argument("--timestamp", default="2025-04-01T00:00:00Z")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--environment", help="Descrição da infraestrutura da API e do Redis")
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0:
        parser.error("use --requests >= 1 e --warmup >= 0")

    transaction = {
        "account_id": args.account_id,
        "amount": 9000,
        "lat": 35.68,
        "lon": 139.69,
        "timestamp": args.timestamp,
    }
    payload = json.dumps(transaction).encode()
    connection = HTTPConnection(args.host, args.port, timeout=10)
    try:
        timings = measure(connection, payload, args.requests, args.warmup)
    except (OSError, HTTPException, RuntimeError) as exc:
        parser.exit(1, f"Falha em POST /score: {exc}\n")
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "endpoint": f"http://{args.host}:{args.port}/score",
                "transaction": transaction,
                "requests": args.requests,
                "warmup": args.warmup,
                "concurrency": 1,
                **timings,
                "client": {
                    "os": platform.platform(),
                    "architecture": platform.machine(),
                    "logical_cpus": os.cpu_count(),
                    "python": platform.python_version(),
                },
                "server_environment": args.environment,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
