"""Command-line entry point for the explicit landuse dataset audit."""

from collections.abc import Iterable, Mapping
from typing import cast

from .dataset_audit import audit_rows, write_audit_artifacts
from .dataset_loader import load_streaming_rows

__all__ = ["main"]


def main() -> None:
    """Audit the pinned stream, write derived artifacts, and report readiness."""

    rows = cast(Iterable[Mapping[str, object]], load_streaming_rows())
    result = audit_rows(rows)
    report_path, manifest_path = write_audit_artifacts(result)
    print(f"audit_report: {report_path}")
    print(f"split_manifest: {manifest_path}")
    print(f"readiness: {result.report.ready}")
    if result.report.review_required_reasons:
        print(
            "review_required_reasons: "
            + ", ".join(result.report.review_required_reasons)
        )
        raise SystemExit(2)
