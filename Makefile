IMAGE ?= docker.io/saadamer/flux-endpoint
SLIM_IMAGE ?= docker.io/saadamer/flux-endpoint-slim
IMAGE_INPUTS = Dockerfile .dockerignore pyproject.toml uv.lock LICENSE src
TAG ?= $(shell uv version --short)-$(shell git log -1 --format=%h -- $(IMAGE_INPUTS) 2>/dev/null | grep . || echo dev)
PREVIOUS_BUILD = $(shell git log -2 --format=%h -- $(IMAGE_INPUTS) 2>/dev/null | sed -n 2p)
DOCKER ?= docker
HF_TOKEN_FILE ?= $(HOME)/.cache/huggingface/token
SPEC ?= deploy/endpoint.toml
DEPLOY_ARGS ?=
TINY_FLUX_REPO = hf-internal-testing/tiny-flux-pipe
TINY_FLUX_REVISION = a98bc4ec6a80c47c477ed22d7e81c79872d9c723
LOCAL_PORT ?= 8010

.PHONY: install check lint typecheck test audit image image-slim deploy deploy-slim rollback \
	smoke teardown worker-local

install:
	uv sync --all-extras

check: lint typecheck test

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run flake8

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

audit:
	uv export --frozen --all-extras --all-groups --no-emit-project --quiet --format requirements-txt \
		| uv run pip-audit --disable-pip --requirement /dev/stdin

image:
	$(DOCKER) buildx build --platform linux/amd64 \
		--secret id=hf_token,src=$(HF_TOKEN_FILE) \
		--tag $(IMAGE):$(TAG) --push .

image-slim:
	$(DOCKER) buildx build --platform linux/amd64 --build-arg MODEL=none \
		--tag $(SLIM_IMAGE):$(TAG) --push .

deploy:
	uv run flux-endpoint deploy $(SPEC) --image $(IMAGE):$(TAG) $(DEPLOY_ARGS)

deploy-slim: SPEC = deploy/cached-model.toml
deploy-slim:
	uv run flux-endpoint deploy $(SPEC) --image $(SLIM_IMAGE):$(TAG) $(DEPLOY_ARGS)

rollback:
	@test -n "$(PREVIOUS_BUILD)" || { echo "No earlier build in the git history." >&2; exit 1; }
	uv run flux-endpoint deploy $(SPEC) --image $(IMAGE):$(shell uv version --short)-$(PREVIOUS_BUILD) $(DEPLOY_ARGS)

smoke:
	uv run pytest -m live

teardown:
	uv run flux-endpoint teardown $(SPEC)

worker-local:
	FLUX_MODEL_DIR=$$(uv run hf download $(TINY_FLUX_REPO) --revision $(TINY_FLUX_REVISION) --quiet) \
	FLUX_DEVICE=cpu FLUX_DTYPE=float32 FLUX_MAX_SEQUENCE_LENGTH=48 FLUX_MIN_GPU_MEMORY_GIB=0 RUNPOD_LOG_LEVEL=INFO \
	uv run python -m flux_endpoint.worker.handler --rp_serve_api --rp_api_port $(LOCAL_PORT)
