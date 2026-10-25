# Architecture

## Components

| Component | Where it runs | What it does |
|---|---|---|
| Worker image | Docker Hub, private repository | Holds the handler, the Python environment and the FLUX.1-dev weights |
| Serverless endpoint | Runpod | Queues jobs, starts and stops GPU workers, and keeps the results |
| Worker | A Runpod GPU, 48 GB or more | Loads the model once, then turns each job into an image |
| `flux-endpoint` CLI | Your machine | Deploys, calls, benchmarks and operates the endpoint |
| Runpod REST API v2 | `api.runpod.io` | Control plane: endpoints, registry credentials, workers, logs and GPU prices |
| Runpod job API | `api.runpod.ai` | Data plane: submit jobs, read their status, cancel them and read queue health |
| Output bucket | Any S3-compatible store | Optional. Receives the images when a client wants URLs instead of base64 |

## Three stages

Each stage is one command.

### 1. Build the image: `make image`

![Build flow: weights stage, runtime stage and private Docker Hub repository](images/build.svg)

- The `weights` stage downloads FLUX.1-dev from Hugging Face at a pinned revision, on the
  build machine's own platform. The Hugging Face token reaches it only as a BuildKit secret
  and never enters a layer.
- The `model-baked` stage puts the weights on the base image, with one layer per transformer
  shard and one for the two T5 shards. Because the weights sit below the code, a code release
  leaves them unchanged, and a host that already has an earlier release pulls only the small
  layers on top.
- The `runtime` stage installs the locked Python dependencies and the worker code on top. The
  worker runs as the non-root user `worker`.
- `make image` pushes the result under an immutable tag, `<version>-<git commit>`, where the
  commit is the last one that changed the image inputs.
- `make image-slim` builds the same stages on top of the empty `model-none` stage and pushes
  the result to a public repository. It can be public because it holds no weights and no
  secrets.

### 2. Deploy the endpoint: `make deploy`

![Deploy flow: find the endpoint by name, add the registry credential, create or update, read back](images/deploy.svg)

The deploy reads `deploy/endpoint.toml` and calls the Runpod REST API v2:

1. Look up the endpoint by name. Two endpoints with the same name stop the deploy.
2. Add the registry credential that lets workers pull the private image, if it is missing.
3. Create the endpoint, or update the one that exists.
4. Read the endpoint back and check every field the deploy sent. A setting that did not apply
   fails the deploy with the name of the field.

### 3. Run a job: `flux-endpoint generate`

![Run flow: client, job API, queue and GPU worker](images/run.svg)

1. The client posts the request to `/run` with a time to live, after which an abandoned job
   expires.
2. The job waits in the queue. If no worker is running, Runpod starts one right away. When
   jobs wait longer than 4 seconds, the scaler adds more workers, up to 2.
3. A worker takes the job. On a cold start it loads the model first, and on a new host Runpod
   pulls the image before that.
4. The worker returns the result to the job API, either the image as base64 or a URL in the
   output bucket.
5. The client reads the result from `/status`. If the client stops waiting for any reason, it
   cancels the job.
6. With an output bucket, the client downloads the image over https.

## Inside a worker

![Worker lifecycle: startup once, then validate, generate, encode and return each job](images/worker.svg)

A worker does its startup work once:

1. Settings come from the `FLUX_*` environment variables. If an output bucket is set without
   its credentials, the worker stops here.
2. The GPU must have at least 40 GiB of memory. With less, the worker stops and says why.
3. FLUX.1-dev loads in bf16 straight onto the GPU, from safetensors files only.
4. Runpod runs a fitness check, one step at 256 x 256. A worker that fails it never takes a
   job.

Then each job takes the same path:

1. Validate the input. Unknown fields and values outside the limits fail the job before any
   GPU work.
2. Resolve the seed. A seed of -1 becomes a random seed, and the result reports the seed used.
3. Generate the image. The noise comes from a CPU generator, so a seed gives the same starting noise on
   every GPU type.
4. Encode the image as PNG or JPEG.
5. Return it inline, or upload it and return its URL, with the seed, the GPU name and the
   timings.

A CUDA out-of-memory error fails the job and restarts the worker, so the next job starts on a
clean GPU.

## Where the model comes from

The same worker can load the weights from three places. `FLUX_MODEL_DIR` sets the folder it
loads from.

| Delivery | `FLUX_MODEL_DIR` | Image |
|---|---|---|
| Baked into the image (default) | `/models/flux-1-dev` | `flux-endpoint`, about 30 GB, private |
| Runpod cached model | `/runpod-volume/huggingface-cache/hub/models--black-forest-labs--FLUX.1-dev` | `flux-endpoint-slim`, public |
| Global Volume | `/runpod-volume/flux-1-dev` | `flux-endpoint-slim`, public |

For a cached model, the worker follows `refs/main` in the Hugging Face cache folder to find the
right snapshot. See [Design](design.md#model-delivery) for the trade-offs and
[Results](results.md) for the cold starts of all three.

## Code map

All the code is in one package, `flux_endpoint`, split into modules by task:

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

`schema.py` is the one module that the worker and the clients share. The worker image installs
the `worker` extra and nothing else, and a client needs only the `client` extra. For
development, `make install` installs both.

## State and data

| Data | Where it lives | How long |
|---|---|---|
| Weights | The image, a cached model or a Global Volume | Until the next release |
| Job input and result | Runpod's queue | Up to 30 minutes after the job completes. See [API](api.md#routes) |
| Images in the output bucket | Your bucket | The presigned URL works for 7 days |
| Prompts | Only in the job input | Same as the job input. The worker never logs them |
| Benchmark samples | `outputs/` on your machine | Until you delete them |
