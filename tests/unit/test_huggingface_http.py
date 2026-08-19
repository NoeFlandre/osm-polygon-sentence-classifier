from typing import Any

import httpx

from osm_polygon_sentence_classifier import huggingface_http


def test_huggingface_client_factory_forces_ipv4_and_bounds_connect_time(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def capture(factory: Any) -> None:
        captured["factory"] = factory

    monkeypatch.setattr(huggingface_http, "_install_client_factory", capture)
    monkeypatch.setattr(huggingface_http, "_configured", False)

    huggingface_http.configure_huggingface_http()

    client = captured["factory"]()
    try:
        assert isinstance(client, httpx.Client)
        assert client.timeout.connect == huggingface_http.CONNECT_TIMEOUT_SECONDS
        assert client.timeout.read is None
        assert client._transport._pool._local_address == "0.0.0.0"
    finally:
        client.close()


def test_huggingface_http_configuration_is_installed_once(monkeypatch) -> None:
    calls = 0

    def capture(_factory: Any) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(huggingface_http, "_install_client_factory", capture)
    monkeypatch.setattr(huggingface_http, "_configured", False)

    huggingface_http.configure_huggingface_http()
    huggingface_http.configure_huggingface_http()

    assert calls == 1
