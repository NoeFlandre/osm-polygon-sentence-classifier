"""Run the test suite and enforce a CRAP score below six per function."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from coverage import Coverage
from radon.complexity import cc_visit
from radon.visitors import Function

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "osm_polygon_sentence_classifier"
MAX_CRAP = 6.0


def _run_tests(data_file: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--branch",
        f"--data-file={data_file}",
        f"--source={SOURCE_ROOT}",
        "-m",
        "pytest",
        "-q",
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _coverage_lines(
    coverage: Coverage,
    path: Path,
) -> tuple[set[int], set[int]]:
    _, statements, _, missing, _ = coverage.analysis2(str(path))
    statement_lines = set(statements)
    return statement_lines, statement_lines - set(missing)


def _crap_score(complexity: int, coverage_fraction: float) -> float:
    return complexity**2 * (1 - coverage_fraction) ** 3 + complexity


def _scores(coverage: Coverage) -> list[tuple[float, Path, Function, float]]:
    scores: list[tuple[float, Path, Function, float]] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        statement_lines, covered_lines = _coverage_lines(coverage, path)
        for block in cc_visit(path.read_text(encoding="utf-8")):
            if not isinstance(block, Function):
                continue
            lines = {
                line
                for line in statement_lines
                if block.lineno <= line <= block.endline
            }
            if not lines:
                continue
            coverage_fraction = len(lines & covered_lines) / len(lines)
            score = _crap_score(block.complexity, coverage_fraction)
            scores.append((score, path, block, coverage_fraction))
    return sorted(scores, reverse=True, key=lambda item: item[0])


def _report(scores: list[tuple[float, Path, Function, float]]) -> int:
    failures = [score for score in scores if score[0] >= MAX_CRAP]
    for score, path, block, coverage_fraction in failures:
        print(
            f"CRAP {score:.2f} >= {MAX_CRAP:.0f}: "
            f"{path.relative_to(ROOT)}:{block.lineno} "
            f"{block.fullname} (complexity={block.complexity}, "
            f"coverage={coverage_fraction:.1%})"
        )
    if not failures:
        print(f"CRAP passed: {len(scores)} functions below {MAX_CRAP:.0f}")
    return int(bool(failures))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sentence-classifier-crap-") as directory:
        data_file = Path(directory) / ".coverage"
        if _run_tests(data_file):
            return 1
        coverage = Coverage(data_file=str(data_file))
        coverage.load()
        return _report(_scores(coverage))


if __name__ == "__main__":
    raise SystemExit(main())
