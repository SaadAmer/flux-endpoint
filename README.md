# FLUX.1-dev on Runpod Serverless

A Runpod Serverless endpoint that turns a text prompt into an image with
[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev). The model is baked into the
Docker image. One Python package holds the worker, the clients, deployment as code and a
benchmark, behind one command: `flux-endpoint`.

The Runpod handler is [`src/flux_endpoint/worker/handler.py`](src/flux_endpoint/worker/handler.py).
The image is built from the [`Dockerfile`](Dockerfile), and
[`deploy/endpoint.toml`](deploy/endpoint.toml) holds the endpoint settings.

## What it does

- Serves FLUX.1-dev on Runpod Serverless. A prompt goes in, and an image comes back as base64
  or as a presigned URL.
- Accepts the same request fields as Runpod's FLUX.1-dev Public Endpoint. A client that sends
  `guidance` can move between the two, and the CLI reads the image field of either.
- Deploys the endpoint as code on the Runpod REST API v2, and checks every setting after the
  write.
- Measures latency and cost per image on each GPU type, and the volume at which a dedicated
  worker becomes cheaper than the Public Endpoint.
- Compares cold starts for three ways to deliver the model: in the image, as a Runpod cached
  model, and on a Global Volume.
- Uses production controls: a non-root worker, pinned and audited dependencies, retries with
  backoff and a time limit on every job. The tests cover every line and branch, and CI
  requires 90%.

## Results

| | |
|---|---|
| ![A lighthouse on a rocky coast at dusk, with waves breaking on the rocks](docs/images/demo/lighthouse.jpg) | ![A red fox standing in fresh snow, lit from behind by the morning sun](docs/images/demo/fox.jpg) |
| ![An isometric illustration of server racks glowing purple on a dark floor](docs/images/demo/data-center.jpg) | ![A steaming bowl of ramen on a wooden table by a window](docs/images/demo/ramen.jpg) |

Measured on 19 October 2026:

- An L40S made a 1024 x 1024 image with 28 steps in 15.7 seconds for $0.0076, and an A100 in
  12.8 seconds for $0.0096.
- The Public Endpoint charged $0.012 per image and queued for 11.2 seconds at the median, with
  5 jobs in flight. At the flex price, a dedicated GPU was cheaper only above 63% utilization,
  but a warm worker took a job within 0.4 seconds.
- A full cold start took 22 to 24 seconds at the median with the model in the image or as a
  Runpod cached model, and 6 minutes from a Global Volume. FlashBoot cut 3 of 5 restarts to
  about 1 second.

[Results](docs/results.md) has every GPU type, the scale-out test, the cold starts and the
platform findings.

## Try it

There are four ways to run it, from a laptop with no account to a full build.
[Try it](docs/try-it.md) walks through each one.

| Way | Needs |
|---|---|
| On a laptop, with a tiny test model on the CPU | [uv](https://docs.astral.sh/uv/) and jq |
| Call the live endpoint, with Postman, the CLI or curl | An API key and endpoint ID from the endpoint owner |
| Deploy in your own Runpod account, from the public slim image and a Runpod cached model | A Runpod API key and a Hugging Face token |
| Build everything from source | Also a private container registry |

## Quick start

Run the worker on a laptop, with a tiny test model on the CPU. It needs
[uv](https://docs.astral.sh/uv/) 0.12 or later and jq, on Linux or on a Mac with Apple silicon
and macOS 14 or later. No account is needed:

```bash
git clone https://github.com/SaadAmer/flux-endpoint.git
```

```bash
cd flux-endpoint && make install
```

The install downloads PyTorch and the other worker libraries, about 230 MB on a Mac and about
3.1 GB on Linux. To call an endpoint without running the worker, `uv sync --extra client` is
enough, as [Try it](docs/try-it.md) shows.

```bash
make worker-local
```

When the worker prints `Uvicorn running on http://localhost:8010`, send a job from a second
terminal in the same folder. The tiny model has random weights, so the image is noise:

```bash
curl -s -X POST http://localhost:8010/runsync -H "Content-Type: application/json" -d @test_input.json | jq -r '.output.image_base64' | base64 --decode > local.png
```

## Deploy on Runpod

Way 3 of [Try it](docs/try-it.md) deploys the public slim image in your own Runpod account,
without a build. To build the image with the weights and deploy it, you need Docker with
buildx, a Hugging Face token that can read FLUX.1-dev in `~/.cache/huggingface/token`, and a
private Docker Hub repository. Create the repository as private before the first push, because
a push to a missing repository creates it with the account's default visibility, and the
weights must not be published. Run `docker login`, set `RUNPOD_API_KEY`, `DOCKERHUB_USERNAME`
and a read-only `DOCKERHUB_TOKEN`, then point the build at your repository:

```bash
export IMAGE="docker.io/$DOCKERHUB_USERNAME/flux-endpoint"
```

```bash
make image
```

```bash
make deploy
```

Set `RUNPOD_ENDPOINT_ID` to the ID that the deploy prints, then generate an image:

```bash
uv run flux-endpoint generate "A lighthouse on a rocky coast at dusk, cinematic lighting" --seed 42
```

Remove the endpoint when you finish:

```bash
make teardown
```

[Deployment](docs/deployment.md) covers each step in detail.

## How it works

![The client posts a job, the queue holds it until a GPU worker takes it, and the client reads the result from the job API](docs/images/run.svg)

The client queues a job on the Runpod job API. When jobs wait, the scaler starts a GPU worker.
The worker loads FLUX.1-dev once, then validates each request, generates the image and returns
it with the seed and timings. The client reads the result, and cancels any job it stops
waiting for. [Architecture](docs/architecture.md) explains each stage, from the image build to
the worker lifecycle.

## Documentation

| Guide | Read it to |
|---|---|
| [Try it](docs/try-it.md) | Run the project on a laptop, against the live endpoint or in your own account |
| [Architecture](docs/architecture.md) | See the components, the request path and the worker lifecycle |
| [Deployment](docs/deployment.md) | Build the image, deploy, check, update, roll back and remove the endpoint |
| [API](docs/api.md) | Call the endpoint with curl, Python, the CLI or Postman |
| [Operations](docs/operations.md) | Watch the endpoint and fix common problems |
| [Design](docs/design.md) | See why it is built this way, and how it scales |
| [Security](docs/security.md) | See the threats and the controls against them |
| [Development](docs/development.md) | Set up the project, run the tests and follow the standards |
| [Results](docs/results.md) | Read the measurements, and how to repeat them |

## License

The code is released under the [MIT License](LICENSE). FLUX.1-dev is released under the
[FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
The image with the weights keeps a copy of that license next to them and is pushed only to a
private registry. The public slim image holds no weights.
