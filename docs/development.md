# Development

## Setup

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/). `make install` sets up both
extras, `worker` and `client`, together with the development tools:

```bash
make install
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/flux_endpoint/schema.py` | Request and result contract, shared by the worker and the clients |
| `src/flux_endpoint/worker/` | The Runpod handler, the FLUX pipeline, settings, image encoding and bucket upload |
| `src/flux_endpoint/http.py` | Authenticated HTTP client with retries |
| `src/flux_endpoint/client/` | Job API client, image download and status text |
| `src/flux_endpoint/deploy/` | Deployment spec, REST API v2 client and the deployer |
| `src/flux_endpoint/bench/` | Benchmark, cold-start measurement, statistics, cost model and reports |
| `src/flux_endpoint/cli.py` | The `flux-endpoint` command |
| `tests/` | Mirrors `src/`, plus `integration/` and `live/` |
| `deploy/` | Endpoint specs: `endpoint.toml`, `cached-model.toml`, and the cold-start study in `cold-start/` |
| `postman/` | Postman collection and environment |
| `Dockerfile`, `Makefile` | Image build, and every command this guide uses |
| `docs/` | These guides |

## Checks

```bash
make check
```

`make check` runs ruff format, ruff, flake8, mypy in strict mode, and pytest with a 90%
coverage gate. GitHub Actions runs the same checks on pushes to `main` and on pull requests,
then audits all locked packages for known vulnerabilities. To run the audit locally:

```bash
make audit
```

To run the formatter, the linters and mypy on every commit, install the pre-commit hooks:

```bash
uv run pre-commit install
```

## Tests

Most tests are unit tests. They cover every module, with fakes at three boundaries:
diffusers, the Runpod SDK and HTTP. The integration tests run the real handler and the real
diffusers `FluxPipeline` on the CPU with
[`hf-internal-testing/tiny-flux-pipe`](https://huggingface.co/hf-internal-testing/tiny-flux-pipe),
a 1.3 MB pipeline with random weights. They are skipped when Hugging Face is out of reach.
`uv run pytest` runs both kinds.

The live test sends one job to the deployed endpoint and spends credits. Run it with
`make smoke`, with `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` set.

## Run the worker on a laptop

`make worker-local` starts the real handler with the tiny pipeline on the CPU, behind the
Runpod SDK's local API server on port 8010:

```bash
make worker-local
```

Then send it a job from another terminal:

```bash
curl -s -X POST http://localhost:8010/runsync -H "Content-Type: application/json" -d @test_input.json
```

## Check the image without the full download

The real build downloads 33.7 GB. To check a change to the Dockerfile, replace the `weights`
stage with a local folder that has the same layout and small placeholder files:

1. Create `/tmp/fake-weights/weights/` with `model_index.json`, the `config.json` and
   tokenizer files, and the safetensors files of the FLUX.1-dev layout, for example
   `transformer/diffusion_pytorch_model-00001-of-00003.safetensors`.
2. Make the files readable for every user, because the worker does not run as root:

   ```bash
   chmod -R a+rX /tmp/fake-weights
   ```

3. Build with the folder as the `weights` build context:

   ```bash
   docker buildx build --platform linux/amd64 --build-context weights=/tmp/fake-weights -t flux-endpoint:check .
   ```

To run a real job inside that image on the CPU, mount the Hugging Face cache and point
`FLUX_MODEL_DIR` at the tiny pipeline's snapshot, with `FLUX_DEVICE=cpu`,
`FLUX_DTYPE=float32`, `FLUX_MAX_SEQUENCE_LENGTH=48` and `FLUX_MIN_GPU_MEMORY_GIB=0`.

## Standards

- Write the test first. For a bug fix, start with a test that reproduces the bug.
- Public module-level functions and classes have Google-style docstrings. Methods and short
  private helpers get one only when the name does not say enough.
- Each test has a one-line `Given ..., When ..., Then ...` docstring.
- Comments only give reasons that the code cannot show.
- Functions stay under 30 lines, with guard clauses instead of nested conditions. Lines are at
  most 100 characters.
- Every parameter and return value has a type, and mypy runs in strict mode.
- There is no `noqa` or `type: ignore` anywhere. When a rule does not fit a file,
  `pyproject.toml` relaxes it for that file, with the reason next to it.

## Dependencies

`uv.lock` pins every package with hashes, and uv ignores releases newer than 7 days. To update
a package:

```bash
uv lock --upgrade-package diffusers
```

Dependabot opens a weekly pull request for the lock, the Docker base image digests and the
GitHub Actions. It does not change the Python or uv version of a base image. Change those by
hand, together with `.python-version` and the uv version in `.github/workflows/ci.yml`.

## Releases

The image tag is the package version from `pyproject.toml` plus the last commit that changed
the image inputs, such as the Dockerfile, the lock or `src/`. A commit that changes only docs
or specs keeps the tag. To ship or roll back a release, see
[Deployment](deployment.md#update-and-roll-back).
