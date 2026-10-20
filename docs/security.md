# Security

The endpoint is not public. Every job needs a Runpod API key, so the clients are services of
the account owner. The design still treats requests, dependencies and downloaded files as
untrusted.

## Threats and controls

| Threat | Control |
|---|---|
| A secret leaks from the repository, the image or shell history | No secret is in the repository or the image. The build reads the Hugging Face token as a BuildKit secret. The CLI reads `RUNPOD_API_KEY` only from the environment, and masks the Docker Hub token in API errors. Bucket keys are Runpod secrets, and the registry credential is a read-only Docker Hub token |
| A client API key leaks | Give clients a Restricted key that can reach only this endpoint. `deploy`, `teardown`, `status`, `logs`, `coldstart` and the cost tables of `bench` and `report` call the REST API, so only the operator's key needs that access |
| A job uses too much GPU time | The request limits cap the size, the steps and the prompt length. The endpoint caps workers at 2 and jobs at 600 seconds, and each client job has a time to live |
| A job writes false log lines | Field names from the client are escaped in errors and logs. Prompts never go to the logs |
| A weight file or the worker process is compromised | The worker loads only safetensors files, with Hugging Face offline. The baked image pins the revision. The worker runs as a non-root user and cannot change the code, the Python environment or the baked weights |
| A dependency is compromised | `uv.lock` pins every package with hashes, with a 7-day cooldown on new releases. The download tool in the weights stage resolves as of a fixed date. Base images are pinned by digest, and GitHub Actions by commit. CI checks the lock for known vulnerabilities, and Dependabot proposes updates weekly |
| An image or its URL leaks | The CLI downloads images only over https and without the API key. A presigned URL works for anyone who has it, for 7 days |

## Secrets

| Secret | Where it lives | What reads it |
|---|---|---|
| Hugging Face token | `~/.cache/huggingface/token` on the build machine | The weights stage, through a BuildKit secret mount. It is never a build argument and never part of a layer |
| Runpod API key | The environment of the shell that runs the CLI | Every `flux-endpoint` command. The CLI has no command-line option for it |
| Docker Hub read-only token | The environment, until the first deploy | The deploy, which stores it as a Runpod registry credential |
| Bucket keys | Runpod secrets, set in the endpoint env as `{{ RUNPOD_SECRET_<name> }}` | The worker, through the Runpod SDK |

Use separate Runpod API keys: one with write access for `deploy` and `teardown`, and Restricted
keys that reach only this endpoint for the services that send jobs.

## Worker

- The process runs as the user `worker`, not root. The application code, the Python
  environment and the weights belong to root and are read-only to it.
- The pipeline loads only safetensors files, from local disk, with Hugging Face offline. A
  pickle-based weight file cannot run code.
- uv is mounted only during the build, so it is not in the image.
- Every request is validated before any GPU work, with limits on the prompt length, the image
  size and the number of steps. Unknown fields are rejected.
- Field names that hold control characters are escaped before they reach an error or a log.
- The log level is `INFO`. At `DEBUG`, the Runpod SDK would log every job output, including the
  image.

## Clients

- The image download uses a separate HTTP client without the API key, and accepts only https
  URLs.
- Throttled requests are retried with backoff. A write is retried only after a 429, which the
  API sends before it acts, or when the connection failed before the request went out. So a
  job is never submitted twice.
- Every job carries a time to live, and the client cancels a job when it stops waiting for it.
- Every HTTP call has a timeout. The `logs` stream has no read timeout, because it waits for
  new lines.

## Supply chain

- `uv.lock` pins every Python package with hashes. uv ignores releases newer than 7 days, so a
  compromised release has time to be found before it reaches the lock.
- The weights stage installs its download tool with packages published before a fixed date,
  because that stage holds the Hugging Face token.
- Base images are pinned by digest, GitHub Actions by commit, and the pre-commit hooks by
  commit.
- `make audit` and CI check every locked package for known vulnerabilities with pip-audit.
- Dependabot proposes updates to the lock, the image digests and the actions every week, after
  the same 7-day cooldown. ghcr.io gives no publication dates, so the cooldown does not apply
  to the uv images.
- The CI workflow runs with a read-only token, and the checkout does not keep it.

## Data

| Data | Where it goes | Exposure |
|---|---|---|
| Prompts | The job input only | Runpod keeps the input and the result for 30 minutes after the job completes |
| Images | The job result, or the output bucket | A presigned URL works for anyone who has it, for 7 days, so treat it like a secret |
| Weights | The private image, or the owner's cached model or Global Volume | FLUX.1-dev is gated and licensed for non-commercial use, so no public image holds it. The public slim image holds only the code and its libraries |
