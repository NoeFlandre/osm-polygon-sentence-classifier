# syntax=docker/dockerfile:1.7

# The digest selects the linux/amd64 manifest used by Grid'5000. uv.lock pins
# the Python dependency set; the Debian package set is installed from the
# pinned base image's configured mirror during the image build.
FROM python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49

ARG UV_VERSION=0.11.16

ENV HOME=/home/app \
    PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN UV_CACHE_DIR=/tmp/uv-cache uv sync --locked --no-dev --extra training \
    && rm -rf /tmp/uv-cache \
    && chown --recursive app:app /app /opt/venv

USER app

# A help-only worker command is useful for smoke checks and cannot audit,
# train, publish, or submit anything by default.
CMD ["python", "-m", "osm_polygon_sentence_classifier.grid5000_worker", "--help"]
