# Try it

You can run the project in four ways without changing any code.

| Way | Time | Needs | Cost |
|---|---|---|---|
| [1. On a laptop](#1-on-a-laptop) | 10 minutes | [uv](https://docs.astral.sh/uv/) 0.12 or later and jq, on Linux or a Mac with Apple silicon and macOS 14 or later. No account and no GPU | Free |
| [2. Call the live endpoint](#2-call-the-live-endpoint) | 2 minutes | An API key and an endpoint ID from the endpoint owner | Billed to the endpoint owner |
| [3. Deploy in your own Runpod account](#3-deploy-in-your-own-runpod-account) | 15 minutes, plus the first cold start | A Runpod API key, and a Hugging Face token that can read FLUX.1-dev | About $0.01 per image, plus a few cents for each cold start and idle period |
| [4. Build everything from source](#4-build-everything-from-source) | An hour or more | Also a private container registry | Your Runpod credit |

Start by cloning the repository:

```bash
git clone https://github.com/SaadAmer/flux-endpoint.git
```

Ways 2 and 3 use only the CLI. Install its client libraries, about 30 MB:

```bash
cd flux-endpoint && uv sync --extra client
```

Ways 1 and 4 also run the worker or the tests, so they need PyTorch and the other worker
libraries. The download is about 230 MB on a Mac and about 3.1 GB on Linux, where the PyTorch
packages include CUDA:

```bash
cd flux-endpoint && make install
```

Way 2 with Postman or curl needs no install at all.

## 1. On a laptop

`make worker-local` runs the real handler on the CPU, behind the Runpod SDK's local API
server. The model is
[`hf-internal-testing/tiny-flux-pipe`](https://huggingface.co/hf-internal-testing/tiny-flux-pipe),
a 1.3 MB FLUX pipeline with random weights. The image is noise, but the run checks the request
path, the validation and the response format.

```bash
make worker-local
```

The worker is ready when it prints `Uvicorn running on http://localhost:8010`. On a laptop CPU
a job takes about 30 seconds. In a second terminal, in the same folder, send one and save the
image:

```bash
curl -s -X POST http://localhost:8010/runsync -H "Content-Type: application/json" -d @test_input.json | jq -r '.output.image_base64' | base64 --decode > local.png
```

To see the validation, send a field that does not exist:

```bash
curl -s -X POST http://localhost:8010/runsync -H "Content-Type: application/json" -d '{"input": {"prompt": "a cat", "steps": 4}}'
```

The job fails with `Invalid input: steps: Extra inputs are not permitted`, because the field
is `num_inference_steps`.

The whole test suite also runs the same tiny pipeline through diffusers. To run it:

```bash
make check
```

## 2. Call the live endpoint

The endpoint owner sends you the endpoint ID and an API key that can reach only this endpoint.
After an idle period, the first request waits for a cold start while a worker starts and loads
the model.

With Postman:

1. Import `postman/flux-endpoint.postman_collection.json` and
   `postman/runpod.postman_environment.json`.
2. Select the **Runpod** environment and set `apiKey` and `endpointId`.
3. Send **Generate (runsync)** and open the **Visualize** tab of the response to see the image.

For the **Public Endpoint (runsync)** request you need a key of your own.

With the CLI, set `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` in your shell, then:

```bash
uv run flux-endpoint generate "A lighthouse on a rocky coast at dusk, cinematic lighting" --seed 42
```

The command saves the image under `outputs/` and prints the seed, the queue and run times,
and the GPU. For the same call with curl and Python, see [API](api.md).

## 3. Deploy in your own Runpod account

Way 3 uses the public image `docker.io/saadamer/flux-endpoint-slim`, which holds the worker and
its libraries but no weights. Runpod supplies the weights as a cached model, so you have
nothing to build or upload.

You need:

- A Runpod API key with read and write access to Serverless, set as `RUNPOD_API_KEY`.
- A Hugging Face account that has accepted the
  [FLUX.1-dev license](https://huggingface.co/black-forest-labs/FLUX.1-dev), and a read token.

Steps:

1. Create the endpoint. It has no workers yet, so it costs nothing:

   ```bash
   uv run flux-endpoint deploy deploy/cached-model.toml
   ```

2. In the [Runpod console](https://console.runpod.io/serverless), open the new endpoint
   `flux-1-dev-cached-model`, select **Manage**, then **Edit endpoint**. Set **Model** to
   `black-forest-labs/FLUX.1-dev` with your Hugging Face token, set **Max workers** to 2, and
   save. Do both in one edit, or a worker prepared before the model is set can start without
   it. This step is manual because the REST API has no field for a cached model.
3. Set `RUNPOD_ENDPOINT_ID` to the ID that step 1 printed. Runpod now prepares workers with
   the image and the model, and does not bill for this. In my runs it took 2 to 5 minutes.
   Wait until this command shows a worker as `IDLE`:

   ```bash
   uv run flux-endpoint status
   ```

4. Generate an image:

   ```bash
   uv run flux-endpoint generate "A red fox in fresh snow, early morning light" --seed 7
   ```

5. Delete the endpoint when you finish:

   ```bash
   uv run flux-endpoint teardown deploy/cached-model.toml
   ```

You can also create the same endpoint entirely in the console. Use the Docker image named in
`deploy/cached-model.toml`, a 48 GB GPU and CUDA 13.0 or later. Set the environment variable
`FLUX_MODEL_DIR` to the value in that file, and set **Model** as in step 2.

## 4. Build everything from source

Here you build the image with the weights baked in, push it to your own private registry and
deploy it. The steps are in [Deployment](deployment.md).

## Clean up

Ways 3 and 4 leave an endpoint in your account. Remove it with
`uv run flux-endpoint teardown deploy/cached-model.toml` (way 3) or `make teardown` (way 4),
and after way 4 also delete the image from your registry. To end way 1, stop
`make worker-local` with Ctrl-C. Way 2 leaves nothing to remove.
