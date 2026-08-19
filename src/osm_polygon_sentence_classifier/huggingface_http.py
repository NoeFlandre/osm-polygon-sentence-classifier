"""Bounded, IPv4-first HTTP transport for Hugging Face Hub calls."""

from collections.abc import Callable
from typing import Any

CONNECT_TIMEOUT_SECONDS = 10.0
_configured = False


def _client_factory() -> Any:
    import httpx

    return httpx.Client(
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        follow_redirects=True,
        timeout=httpx.Timeout(None, connect=CONNECT_TIMEOUT_SECONDS),
    )


def _install_client_factory(factory: Callable[[], Any]) -> None:
    from huggingface_hub.utils import set_client_factory

    set_client_factory(factory)


def configure_huggingface_http() -> None:
    """Configure the shared Hub client once for reliable Grid'5000 use.

    Grid'5000 and some macOS networks can resolve Hugging Face over IPv6 while
    routing only IPv4 traffic. The default Hub client then waits indefinitely
    during connection setup. Restricting the local bind address to IPv4 and
    bounding only connection setup avoids that hang without imposing a timeout
    on large model or checkpoint uploads.
    """

    global _configured
    if _configured:
        return
    _install_client_factory(_client_factory)
    _configured = True


__all__ = ["CONNECT_TIMEOUT_SECONDS", "configure_huggingface_http"]
