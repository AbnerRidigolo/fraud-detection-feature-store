from unittest.mock import Mock

import pytest

from fraud_fs import benchmark


def test_benchmark_reports_measured_requests_without_warmup(monkeypatch):
    connection = Mock()
    response = connection.getresponse.return_value
    response.status = 200
    timestamps = iter([0, 4_000_000, 10_000_000, 11_000_000, 20_000_000, 22_000_000])
    monkeypatch.setattr(benchmark, "perf_counter_ns", lambda: next(timestamps))

    timings = benchmark.measure(connection, b"{}", requests=3, warmup=2)

    assert timings == {"p50_ms": 2.0, "p95_ms": 4.0}
    assert connection.request.call_count == response.read.call_count == 5
    assert all(call.args[:2] == ("POST", "/score") for call in connection.request.call_args_list)


def test_benchmark_rejects_http_errors():
    connection = Mock()
    response = connection.getresponse.return_value
    response.status = 503
    response.read.return_value = b"model unavailable"

    with pytest.raises(RuntimeError, match="HTTP 503.*model unavailable"):
        benchmark.measure(connection, b"{}", requests=1, warmup=0)
