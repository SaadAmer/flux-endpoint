# Design

## Scope

The case study asks for a `handler.py`, a Docker image that contains the handler and
FLUX.1-dev, a Serverless endpoint, and a demonstration that a text prompt returns an image.
I also added:

- Requests with the same fields as Runpod's FLUX.1-dev Public Endpoint, so a client can move
  between the two.
- Deployment as code. One command creates, changes or recreates the endpoint.
- A benchmark and a cost model that show when a dedicated endpoint is cheaper than the Public
  Endpoint.
- A cold-start study of three ways to deliver the model to a worker.

## Model delivery

| Option | Image size | Trade-off |
|---|---|---|
| Bake into the image (default) | About 30 GB | What the case study asks for. Reproducible, but each new host pulls the large image once |
| Runpod cached model | Small | Runpod places the Hugging Face repo on hosts before workers start, and does not bill the download. Cold starts matched the baked image. One model per endpoint, set in the console only |
| Global Volume (beta) | Small | One copy for every data center. Every cold start reads the 34 GB of weights from object storage, which took 6 minutes at the median in [Results](results.md#where-the-model-comes-from) |
| Network volume | Small | Full file system and fast writes, but every worker must run in the volume's data center |

Runpod's guidance ranks cached models first for models on Hugging Face. The worker supports the
baked image, a cached model and a Global Volume with no code change. It reads the weights from
`FLUX_MODEL_DIR`, and when that is a Hugging Face cache folder, like the one a cached model
creates, it resolves the folder to the snapshot named in `refs/main`. A cached model matched
the baked image in the cold-start study, but I kept the baked image as the default. The case
study asks for the model in the image, and baking the weights in ties their revision to the
image tag.

The weights are pinned to Hugging Face revision `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`,
which is also the repo's `main`. The image holds only the diffusers files, 33.7 GB of the
57.9 GB repo. The repo also ships a 23.8 GB single-file checkpoint and a separate autoencoder
file, both for other tools.

## Image build

- To keep the 34 GB download out of x86 emulation, the `weights` stage runs on the build
  machine's own platform, which on a Mac is native arm64.
- The weights sit below the dependencies and the code. A code release changes only the top
  layers, which is all that a host with an earlier release has to pull. With `COPY --link`, the
  weight layers stay the same when the base image changes, so a push can skip them.
- One Dockerfile builds both images. A `MODEL` build argument selects `model-baked` for the
  image with the weights, or the empty `model-none` stage for the slim image.
- Splitting the weights into a layer per transformer shard, plus one for both T5 shards, keeps
  every layer under 10 GB. Pushes and pulls run in parallel and resume per layer.
- The base image is `python:3.12-slim`, and no CUDA base image is needed. The PyPI torch wheel
  includes the CUDA 13 libraries, and the NVIDIA container runtime on the host provides the
  driver. The endpoint requires CUDA 13.0 or later to match the wheel.
- uv installs the locked dependencies from a bind mount and is not part of the final image.
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
| Refuse GPUs with less than 40 GiB | The worker fails at startup and says why, instead of running out of memory halfway through loading |
| Run a one-step warm-up as a Runpod fitness check | The SDK runs fitness checks before the first job. A worker that cannot generate is marked unhealthy and replaced, and CUDA is initialized before the first real request |
| Validate with Pydantic and reject unknown fields | Invalid jobs fail before any GPU work, and the error names the field |
| Return the seed used, and draw noise from a CPU generator | The same seed then gives the same starting noise on every GPU type |
| Fail the job and set `refresh_worker` on CUDA out-of-memory | The next job runs on a clean GPU |
| Log the model load time and, for each job, the job ID, size, time and GPU. Leave out the prompt | Load time is the billed part of a cold start. Prompts can contain customer data |
| Set `RUNPOD_LOG_LEVEL=INFO` | Keeps the base64 images out of the endpoint logs |

## GPU selection

These are the Serverless prices from the REST API v2 catalog on 19 October 2026, in USD per
hour:

| Pool | GPUs | Memory | Price |
|---|---|---|---|
| `AMPERE_48` | RTX A6000, A40 | 48 GB | 1.22 |
| `ADA_48_PRO` | L40, L40S, RTX 6000 Ada, a 48 GB MIG slice of the RTX PRO 6000 | 48 GB | 1.75 |
| `AMPERE_80` | A100 | 80 GB | 2.72 |
| `ADA_80_PRO` | H100 | 80 GB | 4.79 |

The bf16 pipeline needs about 34 GB. The 48 GB pools are the cheapest GPUs that hold it
without CPU offload. Runpod treats the pool order as a preference only, and my first deploy
landed entirely on MIG slices in the second pool (see
[Results](results.md#platform-findings)). That is why I measured each GPU type on its own
endpoint, with `excluded_gpu_types` narrowing a pool to one card.

The spec lists `ADA_48_PRO`, then `AMPERE_80`, then `AMPERE_48`, in the order of the measured
cost per image in [Results](results.md). The A40 has the lowest hourly price but comes last,
because it was the slowest. The spec also leaves out the MIG slice, which was slower than the
L40S at the same price.

## Scaling

### Capacity

By Little's law, the number of busy workers is jobs per second multiplied by seconds per job.
Set `workers_max` about 25% above that, to leave room for a new worker to load while the others
serve traffic. For example, 600 images an hour at 14 seconds each keeps 2.3 workers busy, and
`workers_max` should then be 3. The spec ships with 2, which is enough for a demonstration.

### Autoscaling

Image requests already take several seconds, so I chose the queue delay scaler. It adds a
worker when a job waits more than 4 seconds. For steady, latency-sensitive traffic, the request
count scaler or active workers fit better. Active workers remove cold starts, but they bill
around the clock. The cost model gives the break-even traffic for one busy worker against
the Public Endpoint.

### Availability

The endpoint uses three GPU pools and no network volume, so workers can start in any data center
that has stock. A Global Volume would not change this, because unlike a network volume it does
not pin workers to one data center. During a GPU shortage, adding a pool is a one-line change to
the spec. `data_center_ids` can restrict placement for data residency, but that leaves fewer
GPUs to choose from.

### Cold starts

The first start on a host pulls about 30 GB while the worker is still scaled down, and then
the worker loads the model, which is billed. FlashBoot restarts recently used workers faster.
Cold starts show up in the longest queue time, which the benchmark reports. A cached model
removes the large pull. See [Results](results.md) for the cold start of each delivery option.

## Clients

- Both API clients retry a 429 with exponential backoff and honor `Retry-After`, as the
  Runpod documentation recommends. A 502, 503 or 504 is retried only for idempotent methods,
  and that keeps a job from being submitted twice.
- Every job carries a time to live of the client timeout plus 60 seconds. An abandoned job
  expires instead of running hours later. The client also cancels a job when it stops waiting
  for any reason, including a timeout, an HTTP error or Ctrl-C.
- At high volume, clients should pass a `webhook` to `JobClient.submit` instead of polling
  `/status`. [API](api.md#routes) has the rate limit and how long results stay available.

## Outputs

By default, images come back inline as base64. The Runpod SDK warns when a job result exceeds 20
MB, and a 1536 x 1536 PNG stays well below that. With `FLUX_OUTPUT_BUCKET` set, the worker
uploads each image through the SDK's `upload_in_memory_object` and returns a presigned URL
instead, like the Public Endpoint does. The URL is valid for 7 days, and a URL instead of the
image keeps job results small at high volume. The bucket has to be outside Runpod, since
Runpod's S3-compatible API has no presigned URLs. If any of the three `BUCKET_*` variables is
missing, the worker refuses to start. Otherwise the SDK would write the images to the worker's
local disk and return a path that no client can open.

## Operations

Logs record the model load time, and the job ID and timings of every job. The output includes
inference time, encode time and the GPU model, and the benchmark uses them to break cost down
by GPU type. `flux-endpoint status` and `flux-endpoint logs` read the queue, the workers and
their logs through the job API and REST API v2. For symptoms and fixes, see
[Operations](operations.md). To run more than one environment, give each one its own spec file
and endpoint name. An image
tag names one build, and a rollback deploys the previous tag (see
[Deployment](deployment.md#update-and-roll-back)).

## Cost model

- The Public Endpoint lists $0.02 per megapixel, which is $0.021 for a 1024 x 1024 image. On
  19 October 2026 each image cost $0.012, at every size from 512 x 512 to 1536 x 1536. If the
  samples contain Public Endpoint jobs, the report uses the cost those jobs returned.
- A Serverless worker bills per second while it loads the model, runs jobs and waits for the
  idle timeout.
- On a busy worker, one image costs `run seconds x price per hour / 3600`.
- A worker that runs all the time costs less than the Public Endpoint above
  `price per hour / Public Endpoint price per image` images per hour. The report computes this
  at the flex price, and also shows it as the share of the worker's time it must be busy. With
  the
  discount that Runpod's sales team gives on active workers, the break-even is lower.
- Scale-to-zero workers also pay for model loading and the idle timeout on every burst, so for
  sparse traffic the Public Endpoint is cheaper.

On 19 October 2026, a busy L40S took 15.7 seconds per 1024 x 1024 image at 28 steps. At
$1.75 per hour that is $0.0076, about two thirds of the $0.012 that the Public Endpoint
charged. An L40S that runs all the time at the flex price breaks even at about 146 images per
hour, or 63% of its time. The numbers for every GPU type are in [Results](results.md).
