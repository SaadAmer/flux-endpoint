# Operations

How to watch the endpoint, and how to fix it when something goes wrong. Each problem gives the
symptom, the cause and the fix. The commands need `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID`.

## Watch the endpoint

```bash
uv run flux-endpoint status
```

`status` shows the queue, the number of workers in each state, and each worker's status, GPU,
data center, uptime and whether it still runs an older release. The list can hold more
workers than `workers_max`, because Runpod prepares extra workers for traffic spikes. Workers
that are initializing or idle are not billed. The worker states come from the REST API worker
list, because the counts on the job API's `/health` route trail it by about 30 seconds:

```text
Jobs: 2 in queue, 1 in progress, 40 completed, 1 failed, 0 retried
Workers: 1 running, 1 idle, 0 initializing, 0 throttled, 0 unhealthy (version 3)

WORKER     STATUS   GPU          DATA CENTER  UPTIME  STALE
w-running  RUNNING  NVIDIA L40S  EU-RO-1      12m     no
w-old      IDLE     -            -            -       yes
```

`logs` shows a worker's recent log lines, then follows new ones until you press Ctrl-C.
System logs cover the host: the image pull, the start and the exit. Container logs cover the
worker's own output. With one worker, `logs` finds it. With more, it lists their IDs, and you
pass one as the first argument.

```bash
uv run flux-endpoint logs --source system
```

```bash
uv run flux-endpoint logs --source container
```

The worker logs one line when the model is loaded, with the load time, and one line per job
with the job ID, the image size, the time and the GPU. It never logs the prompt.

| Signal | Where | Watch for |
|---|---|---|
| Queue length and age | `status`, the `/health` route | Jobs that wait while no worker starts |
| Worker states | `status` | Workers stuck in `INITIALIZING`, `THROTTLED` or `UNHEALTHY` |
| Model load time | The `Loaded FLUX.1-dev ... in N s` log line | A slow load on some hosts or GPU types |
| Job time | `delayTime` and `executionTime` on each job | Queue time rising with steady traffic, which means `workers_max` is too low |
| Spend | Runpod billing | Spend that grows while few jobs complete |

## Jobs stay in the queue

| What you see | Cause | Fix |
|---|---|---|
| No workers, or every worker is `THROTTLED` | No GPU in the endpoint's pools has stock | Add a pool to `gpu_pools` in the spec and deploy again, or allow more data centers |
| Workers stay `INITIALIZING` for minutes | The first pull of the 30 GB image on a new host | Wait. It is not billed while the worker is scaled down. A job that starts a worker before its image is on the host moves the pull into the billed cold start, so after a deploy, send the first job when a worker is `IDLE` |
| The system log shows an image pull error | The registry credential is wrong, or the tag was never pushed | Check the tag that `make image` pushed. For a bad credential, delete `dockerhub-flux-endpoint` on the **Container Registry Auth** tab of the Runpod Credentials page, then run `make deploy` with a new read-only `DOCKERHUB_TOKEN` |
| The worker count equals `workers_max` | The traffic needs more workers | Raise `workers_max`. The Scaling section of [Design](design.md) shows how to size it |

## Workers are unhealthy or restart

| Log line | Cause | Fix |
|---|---|---|
| `GPU has 24 GiB, but FLUX.1-dev in bfloat16 needs at least 40 GiB` | A pool with GPUs that are too small | Remove that pool from `gpu_pools` |
| `CUDA is not available on this worker.` | A driver fault on the host | Runpod replaces the worker. If it repeats, report the GPU type and data center to Runpod |
| `Fitness check failed: generate_warm_up_image` | The pipeline cannot generate on this host | Read the traceback in the container log. Keep `min_cuda_version` at 13.0 for this torch build |
| `Set BUCKET_SECRET_ACCESS_KEY to upload images` | `FLUX_OUTPUT_BUCKET` is set without all three `BUCKET_*` variables | Add the missing variables as Runpod secrets |
| `Model folder ... does not exist.` | The endpoint has no cached model, or a worker prepared before the **Model** was set started without it, or `FLUX_MODEL_DIR` points at the wrong place | Set the **Model** in the console. If it is set, set **Max workers** to 0 and then back, so Runpod prepares new workers with the model. Check `FLUX_MODEL_DIR` |
| `Model folder ... has no model_index.json, no refs/main and 0 snapshots` | The cached model is not on the host, or `FLUX_MODEL_DIR` points at the wrong folder | Check the endpoint's **Model** setting and `FLUX_MODEL_DIR` |
| `Snapshot ... that .../refs/main names is missing` | The cached model is incomplete on the host, or `FLUX_MODEL_DIR` points at the wrong repo | Check the endpoint's **Model** setting and `FLUX_MODEL_DIR` |

## Jobs fail

| Error | Cause | Fix |
|---|---|---|
| `Invalid input: <field>: <reason>` | The request breaks the API contract. Unknown fields are rejected | Correct the request. [API](api.md) lists every field and limit |
| `GPU ran out of memory. Retry with a smaller image.` | The image is too large for the GPU | Use a smaller size. The worker restarts by itself |
| A Python traceback | A bug, or a fault on the host | Find the job ID in the container log and read the lines before it |

For the SDK's own debug output, set `RUNPOD_LOG_LEVEL=DEBUG` in the endpoint settings. At that
level the SDK logs every job output, including the image, so set it back to `INFO` afterwards.

## The first request is slow

A cold start shows as a long queue time on the first job, and in the `Queue max` column of
the benchmark report. The idle timeout of 60 seconds keeps a worker warm during a session. For
a latency target, set `workers_min` to 1, which bills around the clock. A cached model gives
the same cold start from a 3.3 GB image instead of 30 GB, so a new host is ready sooner. A
Global Volume made each cold start about 6 minutes for this model. [Results](results.md)
compares the three, and [Deployment](deployment.md) shows how to set each one up.

## Old workers still run after a deploy

`status` marks them as stale. Runpod moves workers to a new release gradually, so they are
replaced as they finish. To switch at once, set `workers_max` to 0, wait for the workers to
stop, then set it back.

## Roll back

Deploy the previous image tag, as [Deployment](deployment.md#update-and-roll-back) shows.

## Costs are higher than expected

Every scale-up pays for model loading and the idle timeout, so frequent small bursts cost more
than their run time. Compare the run times from `flux-endpoint report` with the bill. Check for
active workers that bill around the clock, and delete endpoints that are no longer needed with
`flux-endpoint teardown --name`. To stop all spend on an endpoint at once, set `workers_max`
to 0.
