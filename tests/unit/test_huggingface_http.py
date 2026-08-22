import threading
from typing import Any

import httpx
import pytest

from osm_polygon_sentence_classifier import huggingface_http


def test_huggingface_client_factory_forces_ipv4_and_bounds_connect_time(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Transport:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            captured["transport"] = self

    class _Timeout:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs
            captured["timeout"] = (args, kwargs)

    def client(**kwargs: object) -> dict[str, Any]:
        captured["client"] = kwargs
        return kwargs

    monkeypatch.setattr(httpx, "HTTPTransport", _Transport)
    monkeypatch.setattr(httpx, "Timeout", _Timeout)
    monkeypatch.setattr(httpx, "Client", client)

    result = huggingface_http._client_factory()

    assert result["transport"].kwargs == {"local_address": "0.0.0.0"}
    assert result["follow_redirects"] is True
    assert captured["timeout"] == (
        (None,),
        {"connect": huggingface_http.CONNECT_TIMEOUT_SECONDS},
    )


def test_huggingface_http_configuration_is_installed_once(monkeypatch) -> None:
    factories: list[Any] = []

    def capture(factory: Any) -> None:
        factories.append(factory)

    monkeypatch.setattr(huggingface_http, "_install_client_factory", capture)
    monkeypatch.setattr(huggingface_http, "_configured", False)

    huggingface_http.configure_huggingface_http()
    huggingface_http.configure_huggingface_http()

    assert factories == [huggingface_http._client_factory]


def test_install_client_factory_registers_the_given_factory(monkeypatch) -> None:
    from huggingface_hub import utils

    def factory() -> object:
        return object()

    captured: list[Any] = []
    monkeypatch.setattr(utils, "set_client_factory", captured.append)

    huggingface_http._install_client_factory(factory)

    assert captured == [factory]


def test_rate_limit_detection_follows_exception_causes() -> None:
    def raise_wrapped_rate_limit() -> None:
        try:
            raise RuntimeError("429 Too Many Requests")
        except RuntimeError as cause:
            raise RuntimeError("checkpoint publication failed") from cause

    with pytest.raises(RuntimeError) as captured:
        raise_wrapped_rate_limit()
    assert huggingface_http.is_rate_limit_error(captured.value)

    assert not huggingface_http.is_rate_limit_error(RuntimeError("network failed"))


@pytest.mark.parametrize(
    "message",
    [
        "429",
        "RATE LIMIT exceeded",
        "too many requests",
    ],
)
def test_rate_limit_message_detection_is_case_insensitive_for_each_marker(
    message: str,
) -> None:
    assert huggingface_http._is_rate_limit_message(message)


def test_rate_limit_detection_prefers_a_cause_over_a_non_rate_limit_context() -> None:
    outer = RuntimeError("outer")
    cause = RuntimeError("rate limit")
    context = RuntimeError("network")
    outer.__cause__ = cause
    outer.__context__ = context

    assert huggingface_http.is_rate_limit_error(outer)


def test_rate_limit_detection_terminates_on_exception_cycles() -> None:
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__cause__ = first

    result: list[bool] = []

    def run() -> None:
        result.append(huggingface_http.is_rate_limit_error(first))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert result == [False]
