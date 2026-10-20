# Architecture

This guide explains how the system works, from the image build to a generated image. Read it
first. The other guides go deeper into one area each.

## Components

| Component | Where it runs | What it does |
|---|---|---|
| Worker image | Docker Hub, private repository | Holds the handler, the Python environment and the FLUX.1-dev weights |
| Serverless endpoint | Runpod | Queues jobs, starts and stops GPU workers, and stores results for 30 minutes |
| Worker | A Runpod GPU, 48 GB or more | Loads the model once, then turns each job into an image |
| `flux-endpoint` CLI | Your machine | Deploys, calls, benchmarks and operates the endpoint |
| Runpod REST API v2 | `api.runpod.io` | Control plane: endpoints, registry credentials, workers, logs and GPU prices |
| Runpod job API | `api.runpod.ai` | Data plane: submit jobs, read their status, cancel them and read queue health |
| Output bucket | Any S3-compatible store | Optional. Receives the images when a client wants URLs instead of base64 |

## Three stages

Everything happens in three stages, and each stage has one command.

### 1. Build the image: `make image`

![The weights stage downloads FLUX.1-dev, the runtime stage adds the worker, and the image goes to a private Docker Hub repository](images/build.svg)

- The `weights` stage downloads FLUX.1-dev from Hugging Face at a pinned revision. It runs on
  the build machine's own platform, and it sees the Hugging Face token only as a BuildKit
  secret, so the token never enters a layer.
- The `model-baked` stage puts the weights on the base image, with one layer per transformer
  shard and one for the two T5 shards. They sit below the code, so a code release leaves them
  unchanged, and a host that pulled an earlier release pulls only the small layers on top.
- The `runtime` stage installs the locked Python dependencies and the worker code on top. The
  worker runs as the non-root user `worker`.
- `make image` pushes the result as `<version>-<git commit>`, a tag that never moves. The
  commit is the last one that changed the image inputs.
- `make image-slim` builds the same stages on the empty `model-none` stage instead, and pushes
  the result to a public repository, because it holds no weights and no secrets.

### 2. Deploy the endpoint: `make deploy`

![The deploy command finds the endpoint by name, adds the registry credential, creates or updates the endpoint, and reads it back](images/deploy.svg)

The deploy reads `deploy/endpoint.toml` and calls the Runpod REST API v2:

1. It finds the endpoint by name. Two endpoints with the same name stop the deploy.
2. It adds the registry credential that lets workers pull the private image, if it is missing.
3. It creates the endpoint, or updates the one that exists.
4. It reads the endpoint back and compares every field it sent. A setting that did not apply
   fails the deploy and names the field.

### 3. Run a job: `flux-endpoint generate`

![The client posts a job, the queue holds it until a GPU worker takes it, and the client reads the result from the job API](images/run.svg)

1. The client posts the request to `/run`, with a time to live, so an abandoned job expires.
2. The job waits in the queue. When it waits longer than 4 seconds, the scaler adds a worker,
   up to 2.
3. A worker takes the job. On a cold start the worker first loads the model, and on a new host
   Runpod first pulls the image.
4. The worker returns the result to the job API: the image as base64, or a URL in the output
   bucket.
5. The client reads the result from `/status`. If the client stops waiting for any reason, it
   cancels the job.
6. With an output bucket, the client downloads the image over https.

## Inside a worker

![Startup work runs once per worker. Each job is validated, generated, encoded and returned](images/worker.svg)

A worker does its startup work once:

1. It reads its settings from the `FLUX_*` environment variables. If an output bucket is set
   without its credentials, it stops here.
2. It checks that the GPU has at least 40 GiB of memory, and stops with a clear reason if not.
3. It loads FLUX.1-dev in bf16 straight onto the GPU, from safetensors files only.
4. Runpod runs a fitness check: one step at 256 x 256. A worker that fails it never takes a
   job.

Then each job takes the same path:

1. Validate the input. Unknown fields and values outside the limits fail the job before any
   GPU work.
2. Resolve the seed. A seed of -1 becomes a random seed, and the result reports the seed used.
3. Generate. The noise comes from a CPU generator, so a seed gives the same starting noise on
   every GPU type.
4. Encode the image as PNG or JPEG.
5. Return it inline, or upload it and return its URL, with the seed, the GPU name and the
   timings.

A CUDA out-of-memory error fails the job and restarts the worker, so the next job gets a clean
GPU. Logs carry the job ID, the image size, the time and the GPU, never the prompt.

## Where the model comes from

The same worker can load the weights from three places. It reads them from `FLUX_MODEL_DIR`.

| Delivery | `FLUX_MODEL_DIR` | Image | Trade-off |
|---|---|---|---|
| Baked into the image (default) | `/models/flux-1-dev` | `flux-endpoint`, about 30 GB, private | Fully reproducible. Each new host pulls the large image once |
| Runpod cached model | `/runpod-volume/huggingface-cache/hub/models--black-forest-labs--FLUX.1-dev` | `flux-endpoint-slim`, public | Runpod places the model on hosts before workers start. Cold starts matched the baked image. Set in the console only |
| Global Volume | `/runpod-volume/flux-1-dev` | `flux-endpoint-slim`, public | One copy for every data center. A cold start took 6 minutes, because it reads all the weights from object storage |

For a cached model, the worker resolves the Hugging Face cache folder to the snapshot that
`refs/main` names. [Results](results.md) compares the cold starts of the three.

## Code map

One package, `flux_endpoint`, holds the code. Each module has one job.

| Module | What it does | Talks to |
|---|---|---|
| `schema.py` | Request and result contract, with the Public Endpoint field names | Worker and clients |
| `worker/handler.py` | The Runpod handler: validate, generate, encode, deliver | Runpod SDK |
| `worker/pipeline.py` | Loads FLUX.1-dev and maps a request to the diffusers pipeline | diffusers, the GPU |
| `worker/settings.py` | Worker settings from `FLUX_*` variables | Environment |
| `worker/imaging.py`, `worker/storage.py` | Image encoding, and upload to the output bucket | Runpod SDK, bucket |
| `http.py` | Authenticated HTTP client that retries throttled requests with backoff | Both Runpod APIs |
| `client/` | Submit, poll, cancel, health, image download and status text | Job API |
| `deploy/` | TOML spec, REST API v2 client, create or update with a read-back check | REST API v2 |
| `bench/` | Benchmark, cold-start measurement, statistics, cost model and reports | Job API, GPU catalog |
| `cli.py` | The `flux-endpoint` commands | All of the above |

The worker and the clients share only `schema.py`. The worker image installs only the
`worker` extra, and a client needs only the `client` extra. `make install` installs both for
development.

## State and data

| Data | Where it lives | How long |
|---|---|---|
| Weights | The image, a cached model or a Global Volume | Until the next release |
| Job input and result | Runpod's queue | 30 minutes after the job completes, for `/run` |
| Images in the output bucket | Your bucket | The presigned URL works for 7 days |
| Prompts | Only in the job input | The worker never logs them |
| Benchmark samples | `outputs/` on your machine | Until you delete them |
