# Operations

Every command on this page needs `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID`.

## Watch the endpoint

```bash
uv run flux-endpoint status
```

`status` prints the queue, a count of workers per state, and one line per worker with its GPU,
data center, uptime and whether it still runs an older release. Expect more workers than
`workers_max`. Runpod keeps spare workers ready for traffic spikes, and it does not bill a
worker while it initializes or sits idle. The states come from the REST API worker list, not
from the job API's `/health` route, which lags behind it.

Example output:

```text
Jobs: 2 in queue, 1 in progress, 40 completed, 1 failed, 0 retried
Workers: 1 running, 1 idle, 0 initializing, 0 throttled, 0 unhealthy (version 3)

WORKER          STATUS   GPU          DATA CENTER  UPTIME  STALE
k2x7d0p9m1qa4e  RUNNING  NVIDIA L40S  EU-RO-1      12m     no
r8w3n6c1f5tz0b  IDLE     -            -            -       yes
```

`logs` prints a worker's recent log lines and then follows new ones until you press Ctrl-C.
Use `--source system` for host events such as the image pull, the start and the exit, and
`--source container` for the worker's own output. If the endpoint has a single worker, `logs`
picks it. Otherwise it lists the worker IDs, and you pass one as the first argument.

```bash
uv run flux-endpoint logs --source system
```

```bash
uv run flux-endpoint logs --source container
```

When the model finishes loading, the worker logs one line with the load time. After that it
logs one line per job, with the job ID, the image size, the time and the GPU. The prompt never
goes to the log.

To find a problem, check:

- `status`, for jobs that wait while no worker starts, and for workers stuck in
  `INITIALIZING`, `THROTTLED` or `UNHEALTHY`.
- The logs, where the `Loaded FLUX.1-dev ... in N s` line shows a slow load on some hosts or
  GPU types.
- The `delayTime` and `executionTime` of each job. A queue time that rises while traffic
  stays steady means `workers_max` is too low.
- Runpod billing, for spend that grows while few jobs complete.

## Jobs stay in the queue

| What you see | Cause | Fix |
|---|---|---|
| No workers, or every worker is `THROTTLED` | No GPU in the endpoint's pools has stock | Add a pool to `gpu_pools` in the spec and deploy again, or allow more data centers |
| Workers stay `INITIALIZING` for minutes | The first pull of the 30 GB image on a new host | Wait for the pull to finish. Runpod does not bill it while the worker is scaled down. If a job starts the worker before the image is on the host, the pull becomes part of the billed cold start. After a deploy, send the first job once a worker shows `IDLE` |
| The system log shows an image pull error | The registry credential is wrong, or the tag was never pushed | Check the tag that `make image` pushed. For a bad credential, delete `dockerhub-flux-endpoint` on the **Container Registry Auth** tab of the Runpod Credentials page, then run `make deploy` with a new read-only `DOCKERHUB_TOKEN` |
| The number of running workers equals `workers_max` | The traffic needs more workers | Raise `workers_max`. To size it, see the Scaling section of [Design](design.md#scaling) |

## Workers are unhealthy or restart

| Log line | Cause | Fix |
|---|---|---|
| `GPU has 24 GiB, but FLUX.1-dev in bfloat16 needs at least 40 GiB` | A pool with GPUs that are too small | Remove that pool from `gpu_pools` |
| `CUDA is not available on this worker.` | A driver fault on the host | Runpod replaces the worker. If it repeats, report the GPU type and data center to Runpod |
| `Fitness check failed: generate_warm_up_image` | The pipeline cannot generate on this host | Read the traceback in the container log. Keep `min_cuda_version` at 13.0 for this torch build |
| `Set BUCKET_SECRET_ACCESS_KEY to upload images` | `FLUX_OUTPUT_BUCKET` is set without all three `BUCKET_*` variables | Add the missing variables as Runpod secrets |
| `Model folder ... does not exist.` | The endpoint has no cached model, or a worker prepared before the **Model** was set started without it, or `FLUX_MODEL_DIR` points at the wrong folder | Set the **Model** in the console. If it is already set, set **Max workers** to 0 and then back. Runpod then prepares new workers with the model. Check `FLUX_MODEL_DIR` |
| `Model folder ... has no model_index.json, no refs/main and 0 snapshots` | The cached model is not on the host, or `FLUX_MODEL_DIR` points at the wrong folder | Check the endpoint's **Model** setting and `FLUX_MODEL_DIR` |
| `Snapshot ... that .../refs/main names is missing` | The cached model is incomplete on the host, or `FLUX_MODEL_DIR` points at the wrong folder | Check the endpoint's **Model** setting and `FLUX_MODEL_DIR` |

## Jobs fail

| Error | Cause | Fix |
|---|---|---|
| `Invalid input: <field>: <reason>` | The request breaks the API contract. Unknown fields are rejected | Correct the request. Every field and limit is in [API](api.md) |
| `GPU ran out of memory. Retry with a smaller image.` | The image is too large for the GPU | Use a smaller size. The worker restarts by itself |
| A Python traceback | A bug, or a fault on the host | Find the job ID in the container log and read the lines before it |

For the SDK's own debug output, set `RUNPOD_LOG_LEVEL=DEBUG` in the endpoint settings. At that
level the SDK logs every job output, including the image. Set it back to `INFO` when you are
done.

## The first request is slow

A cold start shows up as a long queue time on the first job, and in the `Queue max` column of
the benchmark report. Within a session, the 60 second idle timeout keeps a worker warm. If you
have a latency target, set `workers_min` to 1. That worker bills even when no job runs. On a
new host, a cached model needs only the 3.3 GB slim image instead of 30 GB. A Global Volume is
far slower than both. The numbers for each delivery option are in
[Results](results.md#where-the-model-comes-from).

## Old workers still run after a deploy

`status` marks them as stale, and Runpod replaces them as they finish. To switch right away, set
`workers_max` to 0, wait for the workers to stop, then set it back.

## Roll back

Deploy the previous image tag. The steps are in
[Deployment](deployment.md#update-and-roll-back).

## Costs are higher than expected

Frequent small bursts cost more than their run time alone, because every scale-up also pays
for the model load and the idle timeout. Compare the run times from `flux-endpoint report` with
the bill. Look for active workers, which bill even when no job runs, and delete endpoints you
no longer need with `flux-endpoint teardown --name`. Setting `workers_max` to 0 stops all
spend on an endpoint immediately.
