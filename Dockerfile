# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
ARG MODEL=baked

FROM ghcr.io/astral-sh/uv:0.12.10@sha256:2bb3ebca0a796a155094a27773d290c4b074572e6107f171d88d086682fd2500 AS uv

# Weights are platform-independent, so this stage runs natively on the build machine.
FROM --platform=$BUILDPLATFORM ghcr.io/astral-sh/uv:0.12.10-python3.12-trixie-slim@sha256:4bf11151c225e2a4d60e2a576b67d925660055199389d9a64fbdc02b6f3d43d5 AS weights
ARG HF_HUB_VERSION=2.1.1
# Only packages published before this date run in the stage that holds the token.
ARG EXCLUDE_NEWER=2026-10-07T00:00:00Z
ARG FLUX_REPO=black-forest-labs/FLUX.1-dev
ARG FLUX_REVISION=3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
RUN uv tool install --exclude-newer "${EXCLUDE_NEWER}" "huggingface_hub==${HF_HUB_VERSION}"
# The repo also ships a 24 GB single-file checkpoint that diffusers does not need.
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    hf download "${FLUX_REPO}" --revision "${FLUX_REVISION}" --local-dir /weights \
        --exclude "flux1-dev.safetensors" --exclude "ae.safetensors" \
        --exclude "dev_grid.jpg" --exclude "README.md" --exclude ".gitattributes" \
    && rm -rf /weights/.cache

FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS base
RUN useradd --create-home --uid 1000 worker

# The weights sit below the code, so a host that pulled one release reuses them for the next.
# --link keeps these layers the same when the base image changes. One layer per transformer
# shard, and one for the two T5 shards, keeps each layer below 10 GB.
FROM base AS model-baked
COPY --link --from=weights \
    --exclude=transformer/*.safetensors --exclude=text_encoder_2/*.safetensors \
    /weights/ /models/flux-1-dev/
COPY --link --from=weights /weights/text_encoder_2/*.safetensors /models/flux-1-dev/text_encoder_2/
COPY --link --from=weights /weights/transformer/*-00001-of-00003.safetensors /models/flux-1-dev/transformer/
COPY --link --from=weights /weights/transformer/*-00002-of-00003.safetensors /models/flux-1-dev/transformer/
COPY --link --from=weights /weights/transformer/*-00003-of-00003.safetensors /models/flux-1-dev/transformer/

# A cached model or a Global Volume supplies the weights at FLUX_MODEL_DIR instead.
FROM base AS model-none

FROM model-${MODEL} AS runtime
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock LICENSE ./
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra worker --no-install-project
COPY src/ src/
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra worker --no-editable
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    RUNPOD_LOG_LEVEL=INFO \
    FLUX_MODEL_DIR=/models/flux-1-dev
USER worker
CMD ["python", "-m", "flux_endpoint.worker.handler"]
