# Security

The endpoint is not public. Every job needs a Runpod API key, so every client acts for the
account owner. The design still treats requests, dependencies and downloaded files as
untrusted.

## Threats and controls

| Threat | Control |
|---|---|
| A secret leaks from the repository, the image or shell history | No secret is in the repository or the image. The CLI has no command-line option for the API key, and it masks the Docker Hub token in API errors. See [Secrets](#secrets) for where each one lives |
| A client API key leaks | Give clients a Restricted key that can reach only this endpoint. Only the operator's key needs REST API access, for `deploy`, `teardown`, `status`, `logs`, `coldstart` and the cost tables of `bench` and `report` |
| A job uses too much GPU time | Request limits on the size, the steps and the prompt length, a cap of 2 workers and 600 seconds per job, and a time to live on each client job |
| Log injection through a job | Field names from the client are escaped, and prompts stay out of the logs. See [Worker](#worker) |
| A weight file or the worker process is compromised | Safetensors only, with Hugging Face offline, and the baked image pins the revision. The worker runs as a non-root user. See [Worker](#worker) |
| A dependency is compromised | Locked hashes, a 7-day cooldown, pinned base images and actions, and a vulnerability audit in CI. See [Supply chain](#supply-chain) for the details |
| An image, its URL or the API key leaks on download | The CLI downloads images over https only, with a client that holds no API key. Presigned URLs expire after 7 days |

## Secrets

| Secret | Where it lives | What reads it |
|---|---|---|
| Hugging Face token | `~/.cache/huggingface/token` on the build machine | The weights stage, through a BuildKit secret mount instead of a build argument. It never enters a layer |
| Runpod API key | The environment of the shell that runs the CLI | Every `flux-endpoint` command. The CLI has no command-line option for it |
| Docker Hub read-only token | The environment, until the first deploy | The deploy, which stores it as a Runpod registry credential |
| Bucket keys | Runpod secrets, set in the endpoint env as `{{ RUNPOD_SECRET_<name> }}` | The worker, through the Runpod SDK |

Use separate Runpod API keys. Give `deploy` and `teardown` a key with write access, and give
each service that sends jobs a Restricted key for this endpoint.

## Worker

- The process runs as the user `worker`, not root. Root owns the application code, the Python
  environment and the weights, and they are read-only to the worker.
- The pipeline loads safetensors files only, from local disk, with Hugging Face offline. A
  pickle-based weight file never loads, so it cannot run code.
- uv is mounted for the build steps and is not in the image.
- Every request is validated before any GPU work, with limits on the prompt length, the image
  size and the number of steps. Unknown fields are rejected.
- Field names that hold control characters are escaped before they reach an error or a log.
- The log level is `INFO`. At `DEBUG`, the Runpod SDK would log every job output, including the
  image.

## Clients

- The image download uses a separate HTTP client without the API key, and accepts only https
  URLs.
- Throttled requests are retried with backoff. A write is retried only after a 429, which the
  API sends before it acts, or when the connection failed before the request went out, so a
  retry never submits a job twice.
- Every HTTP call has a timeout, except the read side of the `logs` stream, which waits for
  new lines.

## Supply chain

- `uv.lock` pins every Python package with hashes. uv ignores releases newer than 7 days, so a
  bad release has time to be caught before it reaches the lock.
- The weights stage installs its download tool with packages published before a fixed date,
  because that stage holds the Hugging Face token.
- Base images are pinned by digest. GitHub Actions and the pre-commit hooks are pinned by
  commit.
- `make audit` and CI check every locked package for known vulnerabilities with pip-audit.
- Dependabot proposes updates to the lock, the image digests and the actions every week, after
  the same 7-day cooldown. ghcr.io gives no publication dates, so the cooldown does not apply
  to the uv images.
- The CI workflow runs with a read-only token, and the checkout does not keep it.

## Data

| Data | Where it goes | Exposure |
|---|---|---|
| Prompts | The job input only | Runpod keeps the input and the result for up to 30 minutes after the job completes |
| Images | The job result, or the output bucket | Anyone who has the presigned URL can open the image for 7 days, so treat the URL like a secret |
| Weights | The private image, or the owner's cached model or Global Volume | FLUX.1-dev is gated and licensed for non-commercial use, so no public image holds it. The public slim image has the code and its libraries and nothing else |
