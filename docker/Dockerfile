# Production-style image for every Python service (gateway, agents, MCP servers). One image,
# many entrypoints: Compose selects the service via `command`. Multi-stage, uv-installed,
# non-root, minimal base, no secret ever copied in (the Groq key is mounted at runtime).
#
# syntax=docker/dockerfile:1

# ---- builder: resolve and install dependencies + the project into a venv ----
FROM python:3.12-slim AS builder

# uv, pinned by digest-free tag for local dev; CI pins it explicitly.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install *dependencies* first, without the project, so this layer caches across source edits.
COPY pyproject.toml uv.lock ./
# `observability` is an optional extra, so a bare `uv sync` leaves OpenTelemetry out and tracing
# silently no-ops. The image must carry it or the OTel Collector and Jaeger receive nothing.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra observability

# Now copy the source the wheel needs and install the project itself.
COPY packages/ packages/
COPY apps/vm_api_gateway/ apps/vm_api_gateway/
COPY agents/ agents/
COPY mcp_servers/ mcp_servers/
COPY evaluations/ evaluations/
COPY data/ data/
COPY migrations/ migrations/
COPY alembic.ini ./
# pyproject declares `readme = "README.md"`, and hatchling reads it when building the project —
# without this the project install fails with "Readme file does not exist".
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra observability

# ---- runtime: slim, non-root, venv on PATH ----
FROM python:3.12-slim AS runtime

# A dedicated unprivileged user. The app owns nothing writable except /tmp (Compose adds a
# tmpfs), so the container can run with a read-only root filesystem.
RUN groupadd --system vm && useradd --system --gid vm --home-dir /app --no-create-home vm

WORKDIR /app
COPY --from=builder --chown=vm:vm /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0

USER vm

# Default entrypoint is the gateway; every other service overrides `command` in Compose.
EXPOSE 8000
CMD ["uvicorn", "vm_api_gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
