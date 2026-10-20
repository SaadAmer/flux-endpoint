# Design

Why the system is built this way: the decisions, the trade-offs, and how it scales and what
it costs. [Architecture](architecture.md) describes what the parts do.

## Scope

The case study asks for a `handler.py`, a Docker image that contains the handler and
FLUX.1-dev, a Serverless endpoint, and a demonstration that a text prompt returns an image.
The repository also adds:

- Requests with the same fields as Runpod's FLUX.1-dev Public Endpoint. A client that sends
  `guidance` can move between the two, and the CLI reads the image field of either.
- Deployment as code, so the endpoint can be created, changed or recreated with one command.
- A benchmark and a cost model that show when a dedicated endpoint is cheaper than the Public
  Endpoint.
- A cold-start study of three ways to deliver the model to a worker.

## Model delivery

| Option | Image size | Region lock | Trade-off |
|---|---|---|---|
| Bake into the image (default) | About 30 GB | None | What the case study asks for. Fully reproducible, but each new host pulls the large image once |
| Runpod cached model | Small | None | Runpod places the Hugging Face repo on hosts before workers start, and does not bill the download. Cold starts matched the baked image. One model per endpoint, set in the console only |
| Global Volume (beta) | Small | None | One copy for every data center. Every cold start reads the 34 GB of weights from object storage, which took 6 minutes at the median in [Results](results.md#where-the-model-comes-from) |
| Network volume | Small | One data center | Full file system and fast writes, but every worker must run in that data center |

Runpod's own guidance ranks cached models first for models on Hugging Face. The worker
supports all three without a code change: it reads the weights from `FLUX_MODEL_DIR`, and it
resolves a Hugging Face cache folder, as a cached model creates, to the snapshot that
`refs/main` names. [Results](results.md) measures the cold start of each. A cached model
started as fast as the baked image, from a 3.3 GB image. The baked image stays the default,
because the case study asks for the model in the image, and because it fixes the revision
in a build that the image tag names.

The weights are pinned to Hugging Face revision `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`,
which is also the repo's `main`. The image holds only the diffusers files, 33.7 GB of the
57.9 GB repo. The repo also ships a 23.8 GB single-file checkpoint and a separate autoencoder
file for other tools.

## Image build

- The `weights` stage runs on the build machine's own platform. On a Mac it runs natively on
  arm64, so the 34 GB download does not go through x86 emulation.
- The weights sit below the dependencies and the code. A code release changes only the top
  layers, so a host that pulled an earlier release pulls only those. `COPY --link` also keeps
  the weight layers the same when the base image changes, so a push skips them.
- A `MODEL` build argument selects `model-baked` or the empty `model-none` stage, so the image
  with the weights and the slim image come from one Dockerfile.
- Each transformer shard has its own layer, and the two T5 shards share one, so no layer is
  above 10 GB. Pushes and pulls run in parallel and resume per layer.
- The base image is `python:3.12-slim`. The PyPI torch wheel includes the CUDA 13 libraries,
  and the NVIDIA container runtime on the host provides the driver, so a CUDA base image is
  not needed. The endpoint requires CUDA 13.0 or later to match the wheel.
- uv installs the locked dependencies from a bind mount, so uv is not in the final image.
- Runpod's GitHub integration cannot build this image. Its `docker build` step must finish in
  30 minutes, and its documentation lists no build secrets for the gated download.
- The image goes to a private registry, because the weights are gated and licensed for
  non-commercial use only. Workers pull it with a read-only Docker Hub token that Runpod
  stores as a registry credential.

## Worker

| Decision | Reason |
|---|---|
| Load the pipeline once, before the job loop | Every job reuses the weights already on the GPU |
| Load with `device_map` straight onto the GPU | The 34 GB of weights do not stage in host memory during the billed start |
| Refuse GPUs with less than 40 GiB | A clear error at startup instead of an out-of-memory error halfway through loading |
| Run a one-step warm-up as a Runpod fitness check | The SDK runs fitness checks before the first job. A worker that cannot generate is marked unhealthy and replaced, and CUDA is initialized before the first real request |
| Validate with Pydantic and reject unknown fields | Invalid jobs fail before any GPU work, and the error names the field |
| Return the seed used, and draw noise from a CPU generator | A seed gives the same starting noise on every GPU type, so results can be reproduced |
| Fail the job and set `refresh_worker` on CUDA out-of-memory | The next job runs on a clean GPU |
| Log the model load time, and per job the job ID, size, time and GPU, never the prompt | Load time is the billed part of a cold start. Prompts can contain customer data |
| Set `RUNPOD_LOG_LEVEL=INFO` | At its default DEBUG level, the SDK logs every handler output, which would write each base64 image into the endpoint logs |

## GPU selection

Serverless prices from the REST API v2 catalog on 19 October 2026, in USD per hour:

| Pool | GPUs | Memory | Price |
|---|---|---|---|
| `AMPERE_48` | RTX A6000, A40 | 48 GB | 1.22 |
| `ADA_48_PRO` | L40, L40S, RTX 6000 Ada, a 48 GB MIG slice of the RTX PRO 6000 | 48 GB | 1.75 |
| `AMPERE_80` | A100 | 80 GB | 2.72 |
| `ADA_80_PRO` | H100 | 80 GB | 4.79 |

The bf16 pipeline needs about 34 GB, so the 48 GB pools are the cheapest GPUs that hold it
without CPU offload. The pool order is only a preference. On 19 October 2026, with A40 stock
reported as high, Runpod placed every worker of the first deploy on 48 GB MIG slices of the
RTX PRO 6000 in `ADA_48_PRO`. So each GPU type was measured on its own endpoint, with
`excluded_gpu_types` narrowing a pool to one card.

[Results](results.md) sets the order. The L40S made an image for the least money, the A100
was the fastest, and the A40 was the cheapest per hour but the slowest. The spec therefore
lists `ADA_48_PRO`, then `AMPERE_80`, then `AMPERE_48`, and leaves out the MIG slice, which
was slower than the L40S at the same price.

## Scaling

### Capacity

Little's law gives the number of busy workers: jobs per second multiplied by seconds per
job. Set `workers_max` about 25% above that, so a new worker can load while the others serve
traffic. For example, 600 images an hour at 14 seconds each keeps 2.3 workers busy, so
`workers_max` should be 3. The spec ships with 2, which is enough for a demonstration.

### Autoscaling

The endpoint scales on queue delay, and adds a worker when a job waits more than 4 seconds.
This suits image generation, where a request already takes several seconds. Steady,
latency-sensitive traffic is better served by the request count scaler, or by active
workers. Active workers remove cold starts but bill around the clock. The cost model gives
the volume at which one busy worker becomes cheaper than the Public Endpoint.

### Availability

The endpoint uses three GPU pools and no network volume, so workers can start in any data
center that has stock. A network volume would pin every worker to one data center. A Global
Volume does not. During a GPU shortage, adding a pool is a one-line change to the spec.
`data_center_ids` restricts placement for data residency, at the cost of capacity.

### Cold starts

The first start on a host pulls about 30 GB, while the worker is scaled down. Then the worker
loads the model, which is billed. FlashBoot restarts recently used workers faster. The
benchmark reports the longest queue time, which is where cold starts show. A cached model
removes the large pull, and [Results](results.md) measures how much each delivery option
saves.

### Clients

- Both API clients retry a 429 with exponential backoff and honor `Retry-After`, as the
  Runpod documentation recommends. A 502, 503 or 504 is retried only for idempotent methods,
  so a job is never submitted twice.
- Every job carries a time to live of the client timeout plus 60 seconds, so an abandoned job
  expires instead of running hours later. The client also cancels a job when it stops
  waiting for any reason, including a timeout, an HTTP error or Ctrl-C.
- At high volume, clients should pass a `webhook` to `JobClient.submit` instead of polling
  `/status`. `/run` accepts 1,000 requests per 10 seconds per endpoint, and the limit grows
  with the number of workers. Results stay available for 30 minutes after a job completes.

### Outputs

By default, images come back inline as base64. The Runpod SDK warns when a job result
exceeds 20 MB, and a 1536 x 1536 PNG stays well below that. With `FLUX_OUTPUT_BUCKET` set,
the worker uploads each image through the SDK's `upload_in_memory_object` and returns a
presigned URL that stays valid for 7 days, the same shape as the Public Endpoint. This keeps
job results small at high volume. The bucket sits outside Runpod, because Runpod's
S3-compatible API has no presigned URLs. The worker refuses to start without all three
`BUCKET_*` variables, because the SDK would otherwise write the images to the worker's local
disk and return a path that no client can open.

### Operations

Logs carry the model load time, and the job ID and timings of every job. The output carries
inference time, encode time and the GPU model, so the benchmark can break cost down by GPU
type. `flux-endpoint status` and `flux-endpoint logs` read the queue, the workers and their
logs through the job API and REST API v2, and [Operations](operations.md) maps symptoms to
fixes. Each environment gets its own spec file, with its own endpoint name. Image tags carry
the package version and the git commit, so every build is a new release. Runpod moves workers
to a new release gradually, and a rollback is a deploy of the previous tag.

## Cost model

- The Public Endpoint lists $0.02 per megapixel, which is $0.021 for a 1024 x 1024 image. On
  19 October 2026 each image cost $0.012, at every size from 512 x 512 to 1536 x 1536. The
  report therefore compares against the cost that the Public Endpoint jobs return, when the
  samples include them.
- A Serverless worker bills per second while it runs: model loading, jobs and the idle
  timeout.
- On a busy worker, one image costs `run seconds x price per hour / 3600`.
- A worker that runs all the time costs less than the Public Endpoint above
  `price per hour / Public Endpoint price per image` images per hour. The report also shows
  this as the share of the worker's time it must be busy. It uses the flex price. Runpod sells
  active workers at a discount through its sales team, which lowers the break-even.
- Scale-to-zero workers also pay for model loading and the idle timeout on every burst. Sparse
  traffic therefore favors the Public Endpoint, and steady traffic favors a dedicated
  endpoint.

Measured on 19 October 2026, for one 1024 x 1024 image with 28 steps: a busy L40S took 15.7
seconds, which is $0.0076 at $1.75 per hour, about two thirds of the $0.012 that the Public
Endpoint charged. An L40S that runs all the time at the flex price breaks even at about 146
images per hour, or 63% of its time. [Results](results.md) has every GPU type.

## Security

[Security](security.md) lists the threats and the controls against each.
