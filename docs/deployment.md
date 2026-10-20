# Deployment

How to build the image, deploy the endpoint, check it, update it, roll it back and remove it.
[Architecture](architecture.md) explains what each step does inside.

## Before you start

You need:

- A Runpod account with credit, and an API key with read and write access to Serverless.
- A Hugging Face account that has accepted the FLUX.1-dev license, and a read token in
  `~/.cache/huggingface/token`. `hf auth login` writes it there.
- A private Docker Hub repository, because the image contains gated weights under a
  non-commercial license. You push with your own Docker login. Runpod pulls with a read-only
  access token.
- Docker with buildx. On Apple silicon the build uses Rosetta or QEMU for `linux/amd64`.
- [uv](https://docs.astral.sh/uv/).

The commands read these environment variables:

| Variable | Used by | Value |
|---|---|---|
| `RUNPOD_API_KEY` | Every `flux-endpoint` command | A Runpod API key |
| `RUNPOD_ENDPOINT_ID` | `generate`, `bench`, `coldstart`, `status`, `logs` and `make smoke` | The ID that `make deploy` prints |
| `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` | `make deploy`, until the registry credential exists | Your Docker Hub user and a read-only access token |
| `IMAGE`, `SLIM_IMAGE` | `make image`, `make deploy`, and their `-slim` forms | Your own repositories. The defaults are the project owner's |

The CLI reads the API key only from the environment, never from a command-line option, so the
key does not end up in shell history. On macOS, the Keychain keeps it out of files too:

```bash
security add-generic-password -a "$USER" -s runpod-api-key -w
```

```bash
export RUNPOD_API_KEY="$(security find-generic-password -a "$USER" -s runpod-api-key -w)"
```

## Build the image

To build into your own private repository, set `IMAGE` and log in to Docker Hub. `make image`
and `make deploy` both read `IMAGE`:

```bash
export IMAGE="docker.io/$DOCKERHUB_USERNAME/flux-endpoint"
```

```bash
docker login
```

Then install the project, build the image and push it:

```bash
make install
```

```bash
make image
```

`make image` builds for `linux/amd64` and pushes the tag `<version>-<git commit>`, where the
commit is the last one that changed the image inputs: the Dockerfile, the dependency lock, the
license or `src/`. A tag names one build and never moves, so every worker of a release runs the
same image, and a change to the docs or the specs keeps the tag. The first
build downloads 33.7 GB of weights and pushes about 30 GB of compressed layers. Later builds
reuse the weight layers, so a code change pushes only the layers that changed.

`make image-slim` builds the same worker without weights and pushes it, with the same tag, to
`SLIM_IMAGE`, by default the public repository `docker.io/saadamer/flux-endpoint-slim`. It
holds no weights and no secrets, so it can be public. The cached model and Global Volume
variants use it, and so does way 3 of [Try it](try-it.md).

## Deploy

```bash
make deploy
```

The deploy reads `deploy/endpoint.toml`, creates the endpoint or updates the one with the same
name, then reads it back and checks every setting. The first run also stores the read-only
Docker Hub token as the registry credential `dockerhub-flux-endpoint`. It prints the endpoint
ID and the URL that takes jobs.

| Setting | Value | Reason |
|---|---|---|
| GPU pools | `ADA_48_PRO` without the MIG slice, then `AMPERE_80`, then `AMPERE_48` | The order of the measured cost per image: L40S, A100, A40. Runpod treats it as a preference, so a worker can land on a later pool. `--exclude-gpu` pins one card |
| Minimum CUDA | 13.0 | The torch wheel in `uv.lock` is built for CUDA 13 |
| Workers | 0 to 2 | Scale to zero when idle, and cap the spend |
| Idle timeout | 60 s | Keeps a worker warm between the requests of one session |
| Queue delay scaler | 4 s | Adds a worker when a job waits in the queue for more than 4 seconds |
| FlashBoot | On | Faster restarts of recently used workers, at no extra cost |
| Execution timeout | 600 s | Stops runaway jobs |
| Container disk | 20 GB | Scratch space for the worker |

`DEPLOY_ARGS` overrides the name, the image, the GPU pools and excluded GPU types without
editing the spec. For example, to deploy a second endpoint that uses only L40S GPUs:

```bash
make deploy DEPLOY_ARGS="--name flux-1-dev-l40s --gpu-pool ADA_48_PRO --exclude-gpu 'NVIDIA L40' --exclude-gpu 'NVIDIA RTX 6000 Ada Generation' --exclude-gpu 'NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb'"
```

## Check it

Set `RUNPOD_ENDPOINT_ID` to the ID that `make deploy` printed. The smoke test sends one small
job and checks that a PNG of the right size comes back:

```bash
make smoke
```

Then generate an image:

```bash
uv run flux-endpoint generate "A lighthouse on a rocky coast at dusk, cinematic lighting" --seed 42
```

The first job waits for a cold start. On a new host, Runpod first pulls the image while the
worker is scaled down, which is not billed. Then the worker starts and loads the model, which
is billed. [Operations](operations.md) shows how to watch this with `status` and `logs`.

## Update and roll back

Commit the change, then build and deploy:

```bash
make image
```

```bash
make deploy
```

The new tag carries the new commit, so the deploy starts a new release. Runpod replaces the
workers gradually, and `flux-endpoint status` marks the old ones as stale. To roll back to the
build before, which `make rollback` finds from the git history:

```bash
make rollback
```

To deploy any earlier build, pass its tag, as `make deploy TAG=` followed by the tag.

## Variants

The model can also come from a Runpod cached model or a Global Volume. Both use the public
`slim` image. `deploy/cached-model.toml` is the production spec for a cached model. The specs
in `deploy/cold-start/` are the study endpoints that [Results](results.md) compares.

### Runpod cached model

`deploy/cached-model.toml` runs the slim image with the production settings of the main
endpoint, except that it starts with no workers. It needs no registry credential, because the
slim image is public:

```bash
make image-slim
```

```bash
make deploy-slim
```

Then, in the Runpod console, open the endpoint, select **Manage**, then **Edit endpoint**. Set
**Model** to `black-forest-labs/FLUX.1-dev` with a Hugging Face token and **Max workers** to 2
in the same edit. A worker prepared before the model is set can start without it, and then it
restarts in a loop while it bills. The REST API has no field for a cached model yet, so this
step is manual. Runpod downloads the model to hosts before workers start, and does not bill
for the download. Once the model is set, change `workers_max` in the spec to 2, so a later
deploy keeps the workers.

### Global Volume

A Global Volume holds one copy of the weights for every data center, mounted at
`/runpod-volume` on every worker. For this model it is the slowest option: a cold start reads
all 33.7 GB from object storage, which took 6 minutes at the median, against 24 seconds with
the weights in the image, as [Results](results.md#where-the-model-comes-from) shows. It suits
smaller models, or data that a worker reads in parts.

The study filled one this way:

1. Create the volume with `POST /v2/global-volumes` and a name. Global Volumes are in beta.
2. Start a CPU Pod from the image with the weights, with the registry credential and the
   volume mounted at `/data`. Give it the command
   `sh -c 'cp -r /models/flux-1-dev /data/ && sleep infinity'`. The volume does not store
   file timestamps, so `cp -a` fails. The copy took about 2 minutes.
3. Delete the Pod. Then deploy a spec with the volume ID in `global_volume_id` and
   `FLUX_MODEL_DIR = "/runpod-volume/flux-1-dev"`, as `deploy/cold-start/global-volume.toml`
   does.

Delete the endpoint before the volume, because a mounted volume cannot be deleted.

## Remove

```bash
make teardown
```

This deletes the endpoint by name. The registry credential and the image stay. To remove
another endpoint, pass its spec:

```bash
make teardown SPEC=deploy/cached-model.toml
```

For an endpoint deployed with `--name`, such as the L40S example above, pass the name:

```bash
uv run flux-endpoint teardown --name flux-1-dev-l40s
```

## Costs and limits

- A worker bills per second while it runs: loading the model, running jobs, and waiting for
  the idle timeout. Image pulls and cached model downloads happen while the worker is scaled
  down and are not billed.
- The largest hourly spend is `workers_max` times the GPU price. With 2 workers it is $3.50 per
  hour on L40S, and $5.44 if every worker falls back to an A100.
- Every job has a 600 second limit, and every job from the CLI has a time to live, so a stuck
  or abandoned job stops billing.
- The spec keeps no active workers, so an idle endpoint costs nothing.
- Runpod's default account limit is $80 of spend per hour.
