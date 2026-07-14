# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_MANAGED_PYTHON=1 \
    PATH="/app/.venv/bin:${PATH}"

# Apt caches persist across builds via BuildKit cache mounts (Compose local cache).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        ffmpeg \
        build-essential \
        libsndfile1 \
        libsox-dev \
        sox \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
# Project installs editable (uv.lock: editable = "."). Compose bind-mounts ./src → /app/src
# so host code changes apply without rebuilding; restart/watch reloads the process.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 7860

ENTRYPOINT ["parakeet-transcribe"]
CMD ["run"]
