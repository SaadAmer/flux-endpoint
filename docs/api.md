# API

Every request needs a Runpod API key as a bearer token. The examples read the key from
`RUNPOD_API_KEY` and the endpoint ID from `RUNPOD_ENDPOINT_ID`.

## Routes

The endpoint uses Runpod's standard queue routes on `https://api.runpod.ai/v2/<endpoint ID>`.

| Route | Method | What it does |
|---|---|---|
| `/run` | POST | Queue a job and return its ID right away |
| `/runsync` | POST | Queue a job and wait for the result. After a long cold start it returns the job ID and status instead |
| `/status/<job ID>` | GET | Read a job's status and, when it is done, its output |
| `/cancel/<job ID>` | POST | Cancel a queued or running job |
| `/health` | GET | Job and worker counts by state. The worker counts trail the REST API worker list by about 30 seconds |

Results stay available for 30 minutes after a job completes. `/run` accepts 1,000 requests
per 10 seconds per endpoint, and the limit grows with the number of workers.

## Request

The job input goes under `input`. The fields are the same as on Runpod's FLUX.1-dev Public
Endpoint.

| Field | Type | Default | Limits |
|---|---|---|---|
| `prompt` | string | required | 1 to 2000 characters |
| `negative_prompt` | string | none | Used only when `true_cfg_scale` is above 1 |
| `width`, `height` | integer | 1024 | 256 to 1536, a multiple of 64 |
| `num_inference_steps` | integer | 28 | 1 to 50 |
| `guidance` | number | 3.5 | 0 to 10 |
| `true_cfg_scale` | number | 1.0 | 1 to 10. Above 1, with a `negative_prompt`, each step runs the transformer twice |
| `seed` | integer | -1 | -1 picks a random seed, or 0 to 4294967295 |
| `image_format` | string | `jpeg` | `png` or `jpeg` |

Unknown fields are rejected, so a misspelled field fails the job right away instead of being
ignored.

A job can also carry a `policy`. `ttl` is the time in milliseconds after which Runpod drops
the job, whether it is queued or running. The CLI sets it to its own timeout plus 60 seconds.

```json
{
  "input": {
    "prompt": "A lighthouse on a rocky coast at dusk, waves breaking, cinematic lighting",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "image_format": "png"
  },
  "policy": { "ttl": 660000 }
}
```

## Response

| Field | Description |
|---|---|
| `image_base64` | The encoded image, when the worker has no output bucket |
| `image_url` | A presigned URL to the image in the output bucket, valid for 7 days |
| `image_format` | `png` or `jpeg` |
| `width`, `height` | Image size in pixels |
| `seed` | The seed used. The same seed and input give the same image on the same GPU type |
| `gpu` | The GPU model that generated the image |
| `inference_ms`, `encode_ms` | Time spent on the worker |

The job API wraps the output with the job ID, `status`, `delayTime` (time in the queue, in
milliseconds) and `executionTime` (time on the worker).

## Errors

| Case | Result |
|---|---|
| Invalid input | Job `FAILED` with `Invalid input: <field>: <reason>`. Nothing runs on the GPU |
| GPU out of memory | Job `FAILED`. The worker restarts, so the next job gets a clean GPU |
| Any other exception | Job `FAILED` with the traceback that the Runpod SDK records |
| Job over 600 seconds | Job `TIMED_OUT` |

## Differences from the Public Endpoint

| | This endpoint | FLUX.1-dev Public Endpoint |
|---|---|---|
| Endpoint ID | Your own | `black-forest-labs-flux-1-dev` |
| `guidance` default | 3.5, the value Black Forest Labs recommends for FLUX.1-dev | 7.5 |
| `true_cfg_scale` | Supported | Not available |
| Image in the output | `image_base64`, or `image_url` with an output bucket | An image URL valid for 7 days, in `result`. Runpod's docs name the field `image_url` |
| Price | Per second of GPU time | Listed at $0.02 per megapixel. Each image cost $0.012 on 19 October 2026, at every size from 512 x 512 to 1536 x 1536 |

Requests without `true_cfg_scale` work on both, but send `guidance` because the defaults
differ. The CLI and the Python client always send it. They can read the image from any of
`image_base64`, `image_url` and `result`.

## Examples

### curl

`/runsync` waits for the image. Pipe the response through `jq` and `base64` to save it to a
file:

```bash
curl -s -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_input.json | jq -r '.output.image_base64' | base64 --decode > lighthouse.png
```

For long jobs, queue with `/run` and keep the job ID:

```bash
JOB_ID="$(curl -s -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_input.json | jq -r '.id')"
```

Read `/status` until the status is `COMPLETED`, then take the image from
`.output.image_base64`. To check the status and the timings:

```bash
curl -s "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/$JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | jq '{status, delayTime, executionTime}'
```

### The CLI

The CLI polls for the result and saves the image under `outputs/`. To generate an image:

```bash
uv run flux-endpoint generate "A lighthouse on a rocky coast at dusk, cinematic lighting" --seed 42
```

The same command works against the Public Endpoint:

```bash
uv run flux-endpoint generate "A lighthouse on a rocky coast at dusk, cinematic lighting" --seed 42 -e black-forest-labs-flux-1-dev
```

### Python

```python
import os

from flux_endpoint.client.jobs import JobClient
from flux_endpoint.http import JOB_API_URL, create_runpod_client
from flux_endpoint.schema import GenerationRequest

request = GenerationRequest(prompt="A red fox in fresh snow", seed=7, image_format="png")
with create_runpod_client(os.environ["RUNPOD_API_KEY"], JOB_API_URL) as http:
    result = JobClient(http, os.environ["RUNPOD_ENDPOINT_ID"]).run(
        request.to_job_input(), timeout_s=600
    )
print(result.status, result.output["seed"], result.output["gpu"])
```

The CLI uses the same `JobClient.run`. It sets a time to live, retries throttled requests,
and cancels the job if the call fails or is interrupted.

### Postman

Import `postman/flux-endpoint.postman_collection.json` and
`postman/runpod.postman_environment.json`, select the **Runpod** environment, and fill in
`apiKey` and `endpointId`. **Generate (runsync)** waits for the image and shows it in the
response's **Visualize** tab. **Submit (run)** saves the job ID for **Status** and **Cancel**.
**Public Endpoint (runsync)** sends the same request to Runpod's FLUX.1-dev Public Endpoint for
comparison.
