# Results

Measurements from 19 October 2026 on Runpod Serverless, with the image
`docker.io/saadamer/flux-endpoint:0.1.0-dev`. Every benchmark job made one 1024 x 1024 image
with 28 steps and guidance 3.5, from the five benchmark prompts with seeds 0 to 19. The raw
samples are in [`docs/data/`](data/), and [How to repeat](#how-to-repeat) gives the commands.
A few numbers come from the CLI output, the job API or a worker log instead, and the text
names that source.

## Demo

| | |
|---|---|
| ![A lighthouse on a rocky coast at dusk, with waves breaking on the rocks](images/demo/lighthouse.jpg) | ![A red fox standing in fresh snow, lit from behind by the morning sun](images/demo/fox.jpg) |
| A lighthouse on a rocky coast at dusk, waves breaking, cinematic lighting. Seed 42 | A red fox in fresh snow, early morning light, wildlife photography. Seed 7 |
| ![An isometric illustration of server racks glowing purple on a dark floor](images/demo/data-center.jpg) | ![A steaming bowl of ramen on a wooden table by a window](images/demo/ramen.jpg) |
| An isometric illustration of a small data center with glowing GPU racks. Seed 3 | A bowl of ramen on a wooden table, steam rising, shallow depth of field. Seed 11 |

Each image took 21.5 to 21.8 seconds on a 48 GB MIG slice of an RTX PRO 6000, with 0.1
seconds in the queue, as the CLI reported.

## Speed and cost per GPU

Each GPU type ran on its own endpoint, with `--gpu-pool` and `--exclude-gpu` pinning one card,
and took 20 jobs one after another. The Public Endpoint took the same 20 jobs, 5 at a time.

| GPU | Pool | $/hour | Run p50 (s) | Run p95 (s) | $/image | Images/hour | Break-even images/hour | Break-even utilization |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L40S | `ADA_48_PRO` | 1.75 | 15.7 | 15.8 | 0.0076 | 230 | 146 | 63% |
| A100 SXM 80 GB | `AMPERE_80` | 2.72 | 12.8 | 12.8 | 0.0096 | 282 | 227 | 80% |
| A40 | `AMPERE_48` | 1.22 | 28.5 | 28.6 | 0.0097 | 126 | 102 | 81% |
| RTX PRO 6000, MIG 2g.48gb | `ADA_48_PRO` | 1.75 | 21.7 | 21.8 | 0.0105 | 166 | 146 | 88% |
| Public Endpoint | | | 15.6 | 18.9 | 0.0120 | | | |

The cost per image is the run time multiplied by the hourly flex price, for a worker that
stays busy. Break-even is the rate at which a worker that runs all the time costs as much as
the Public Endpoint at the $0.012 per image that it charged. Runpod sells active workers at a
discount through its sales team, which lowers the break-even.

- The L40S made an image for the least money. The A100 was the fastest, at about the same
  cost per image as the A40, which is the cheapest per hour.
- At the flex price, a dedicated GPU beat the Public Endpoint on cost only above 63% to 88%
  utilization. Scale to zero adds a cold start and the 60 second idle timeout to every burst,
  so for cost alone, sparse traffic belongs on the Public Endpoint.
- A dedicated endpoint wins on latency and control. A warm worker took a job within 0.4
  seconds at the median, and the request format, model revision and GPU type stay fixed.

## Queue and latency

| Target | Queue p50 (s) | Queue max (s) | Total p95 (s) |
|---|---:|---:|---:|
| L40S | 0.3 | 36.2 | 20.4 |
| A100 SXM 80 GB | 0.1 | 19.5 | 15.7 |
| A40 | 0.4 | 34.5 | 32.6 |
| RTX PRO 6000, MIG 2g.48gb | 0.3 | 1.1 | 24.8 |
| Public Endpoint, 5 in flight | 11.2 | 97.2 | 95.0 |

The queue maximum of each GPU is the cold start of its first job. On the MIG slice that job
came after a FlashBoot restart, so it waited only 1.1 seconds. The total is the wall time
that the client saw, including its polling. Earlier the same day, one Public Endpoint job
waited about 7 minutes in the queue, as the CLI measured.

## Scale-out

A burst of 20 jobs with 8 in flight went to the MIG endpoint, which allows 2 workers.

- The queue delay scaler started the second worker within seconds. After its cold start, it
  took its first job 21 seconds into the burst.
- The 20 images took about 250 seconds. Queue p50 was 67.1 seconds, the maximum 90.0
  seconds, and run p50 21.5 seconds.
- Little's law predicts 8 jobs / (2 workers / 21.5 seconds) = 86 seconds in the system, and
  the median time in the system, queue plus run, was 88.6 seconds. To keep the queue short,
  `workers_max` must cover the jobs in flight, as [Design](design.md#scaling) shows.

## Cold starts

| Step | Time | Billed |
|---|---|---|
| First pull of the 30 GB image on a host, from the worker status | 4.5 minutes in US-PA-1, about 24 minutes in CA-MTL-1 and US-MO-1 | No |
| First pull of the 3.3 GB slim image, from the worker status | About 5 minutes | No |
| First job on a fresh host, after the pull | 19.5 s on A100, 34.5 s on A40 and 36.2 s on L40S, for the first benchmark job. 25.2 s on the MIG slice, for the smoke test job, from the job API | Yes |
| Model load from the image, inside a cold start, from the worker log | 5.1 s on the MIG slice | Yes |
| Fitness checks, including the warm-up image, from the worker log | 1.6 s | Yes |

Each cold-start run waits until the REST API worker list has shown an idle worker and no
running worker for 30 seconds, then times one 512 x 512 job with 4 steps. Without that settle
time, a job sent just after a scale-down can still reach the stopping container and look like
a 1 second cold start. The runs without it are in the `*-unsettled-*` files in [`docs/data/`](data/) and are
not used here.

### Where the model comes from

All variants ran on A40, with FlashBoot off and 1 worker, so every run is a full cold start.

| Delivery | Image | Cold start p50 (s) | Cold start max (s) |
|---|---|---:|---:|
| Runpod cached model | 3.3 GB, public | 21.6 | 38.3 |
| Baked into the image | 30 GB, private | 23.9 | 27.9 |
| Global Volume | 3.3 GB, public | 358.9 | 401.7 |

- A Runpod cached model started as fast as the baked image, from an image a ninth of the size
  that needs no private registry and no build with the gated download. Its first run took
  38.3 seconds, with a model load of 14.6 seconds. The other four took 19.9 to 22.9 seconds,
  and the worker log showed a load of 7.8 seconds.
- The worker log on the Global Volume endpoint shows model loads of 302.8 and 388.7 seconds.
  The volume delivered the 33.7 GB of weights at about 90 to 110 MB/s, and every cold start
  reads them again.
- At $1.22 per hour, a Global Volume cold start cost about $0.12, against $0.008 with the
  weights in the image.
- A Global Volume suits data that many data centers share and that a worker reads in parts.
  For weights that every cold start loads in full, the image or a cached model is faster.

### FlashBoot

The first production deploy, which ran on MIG slices, had FlashBoot on. Five settled cold
starts there all ran on the same worker, as the job API shows:

| Run | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Cold start (s) | 18.9 | 1.1 | 43.0 | 1.1 | 1.2 |

FlashBoot restarted the worker in about 1 second in 3 of the 5 runs. It is best effort: the
other 2 runs paid a full cold start on the same worker. The first job of the MIG benchmark,
4.5 minutes after a scale-down, also started in 1.1 seconds.

## Platform findings

- The pool order is a preference. With A40 stock reported as high, Runpod placed every worker
  of the first deploy on MIG slices in the second pool.
- Runpod prepared 5 workers for an endpoint with `workers_max = 2`. Workers that are
  initializing or idle are not billed.
- The image does not count against the container disk. The 20 GB disk had 19.98 GB free with
  the 30 GB image on the host, as the worker log shows.
- The worker counts on the job API's `/health` route trail the REST API worker list by about
  30 seconds after a scale-down. Its `idle` and `ready` counts are free worker slots, not
  workers. So the cold-start probe and `flux-endpoint status` read the worker list.
- Setting a cached model on an endpoint that already had workers left one of them without the
  model. Runpod reported it as idle, started it for the next job, and it restarted in a loop,
  billed, until Max workers went to 0. Workers prepared after the change logged "model ready"
  about 90 seconds after they started. So `deploy/cached-model.toml` starts with no workers.
- A CPU Pod can mount a Global Volume, which made the fill cheap: $0.12 per hour for 17 minutes.
  The volume does not store file timestamps, so `cp -a` onto it fails with "Operation not
  permitted". `cp -r` works, and a byte-by-byte compare confirmed the copy.
- The FLUX.1-dev Public Endpoint returned its image URL in `output.result`, while its docs name
  the field `image_url`. It charged $0.012 for each of the 20 benchmark images, and for two
  single jobs at 512 x 512 and 1536 x 1536, while its docs list $0.02 per megapixel. It
  accepted the job `policy.ttl` that the client sends.

## How to repeat

Every command needs `RUNPOD_API_KEY`. After each deploy, set `RUNPOD_ENDPOINT_ID` to the ID
that the deploy printed, and wait until this command shows an `IDLE` worker, so the image
pull is not billed:

```bash
uv run flux-endpoint status
```

### Speed and cost

Deploy one endpoint per GPU type. For the L40S:

```bash
make deploy DEPLOY_ARGS="--name flux-1-dev-l40s --gpu-pool ADA_48_PRO --exclude-gpu 'NVIDIA L40' --exclude-gpu 'NVIDIA RTX 6000 Ada Generation' --exclude-gpu 'NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb'"
```

Benchmark it, then the Public Endpoint into the same results file:

```bash
uv run flux-endpoint bench --requests 20 --concurrency 1
```

```bash
uv run flux-endpoint bench --endpoint-id black-forest-labs-flux-1-dev --requests 20 --concurrency 5 --timeout 900
```

Print the tables with today's GPU prices. When the samples include Public Endpoint jobs, the
break-even uses the cost that those jobs returned:

```bash
uv run flux-endpoint report
```

### Scale-out

Send the burst to an endpoint that allows 2 workers, into its own results file:

```bash
uv run flux-endpoint bench --requests 20 --concurrency 8 --results outputs/burst.jsonl
```

### Cold starts

Deploy the study endpoint of one delivery method. It runs on A40, with FlashBoot off and 1
worker:

```bash
make deploy SPEC=deploy/cold-start/baked.toml
```

The cached model study runs the slim image, and needs the console step from
[Deployment](deployment.md#runpod-cached-model). The Global Volume study needs a volume of
your own, as the same page shows:

```bash
make deploy-slim SPEC=deploy/cold-start/cached-model.toml
```

Then measure five cold starts. Each run first waits for 30 seconds of scale-down:

```bash
uv run flux-endpoint coldstart --variant baked --runs 5
```

For FlashBoot, run the same command against an endpoint with FlashBoot on, such as the one
that `make deploy` creates, with another `--variant` name.

### Clean up

Remove each endpoint when it is done, by name or by spec:

```bash
uv run flux-endpoint teardown --name flux-1-dev-l40s
```

```bash
make teardown SPEC=deploy/cold-start/baked.toml
```
