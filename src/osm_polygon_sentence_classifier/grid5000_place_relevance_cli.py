"""CLI entry point for the worldwide V2 place-relevance training task."""

from __future__ import annotations

from collections.abc import Sequence

from .grid5000_cli import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guarded Grid'5000 controller for worldwide V2."""

    return _main(argv, task_name="place-relevance-v2")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
