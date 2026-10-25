# Deployment

## Before you start

You need:

- A Runpod account with credit, and an API key that can read and write Serverless endpoints.
- A read token in `~/.cache/huggingface/token` from a Hugging Face account that has accepted
  the FLUX.1-dev license. `hf auth login` writes it there.
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
| `IMAGE`, `SLIM_IMAGE` | `make image`, `make deploy`, and their `-slim` forms | Your own repositories. The defaults are mine |

The CLI reads the API key from the environment only. On macOS you can keep it in the Keychain
instead of a file:

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

`make image` builds for `linux/amd64` and pushes the tag `<version>-<git commit>`. The commit is
the last one that changed the image inputs, which are the Dockerfile, `.dockerignore`,
`pyproject.toml`, the dependency lock, the license and `src/`. Tags are immutable, so every
worker of a release runs the same image. A change to the docs or the specs keeps the tag. The
first build downloads 33.7 GB of weights and pushes about 30 GB of compressed layers. After
that, builds reuse the weight layers, and a code change pushes only the layers that changed.

`make image-slim` builds the same worker and pushes it under the same tag to `SLIM_IMAGE`,
which by default is the public repository `docker.io/saadamer/flux-endpoint-slim`. With no
weights and no secrets inside, it is safe to make public. The cached model and Global Volume
variants use it, and so does way 3 of [Try it](try-it.md).

## Deploy

```bash
make deploy
```

The deploy reads `deploy/endpoint.toml`, creates the endpoint or updates the one with the same
name, then reads it back and checks every setting. The first run also stores the read-only
Docker Hub token as the registry credential `dockerhub-flux-endpoint`. It prints the endpoint
ID and the run URL.

| Setting | Value | Reason |
|---|---|---|
| GPU pools | `ADA_48_PRO` without the MIG slice, then `AMPERE_80`, then `AMPERE_48` | Ordered by the measured cost per image of the L40S, A100 and A40. Runpod treats the order as a preference, and a worker can land on a later pool. `--exclude-gpu` pins one card |
| Minimum CUDA | 13.0 | The torch wheel in `uv.lock` is built for CUDA 13 |
| Workers | 0 to 2 | Scale to zero when idle, and cap the spend |
| Idle timeout | 60 s | Keeps a worker warm between the requests of one session |
| Queue delay scaler | 4 s | The first job starts a worker right away. Jobs that wait more than 4 seconds add more. [Design](design.md#autoscaling) explains the choice |
| FlashBoot | On | Faster restarts of recently used workers, at no extra cost |
| Execution timeout | 600 s | Stops runaway jobs |
| Container disk | 20 GB | Scratch space for the worker |

`DEPLOY_ARGS` overrides the name, the image, the GPU pools and excluded GPU types without
editing the spec. For example, to deploy a second endpoint that uses only L40S GPUs:

```bash
make deploy DEPLOY_ARGS="--name flux-1-dev-l40s --gpu-pool ADA_48_PRO --exclude-gpu 'NVIDIA L40' --exclude-gpu 'NVIDIA RTX 6000 Ada Generation' --exclude-gpu 'NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb'"
```

## Check it

Set `RUNPOD_ENDPOINT_ID` to the ID that `make deploy` printed. On a new host, Runpod first
pulls the image while the worker is scaled down, and does not bill the pull. A job that starts
a worker before the image is on the host pays for the pull as part of its cold start, so wait
until a worker shows as `IDLE`:

```bash
uv run flux-endpoint status
```

The smoke test sends one small job and checks that a PNG of the right size comes back:

```bash
make smoke
```

Then generate an image:

```bash
uv run flux-endpoint generate "A lighthouse on a rocky coast at dusk, cinematic lighting" --seed 42
```

The first job waits for a cold start, and billing begins when the worker starts and loads the
model. To watch this with `status` and `logs`, see [Operations](operations.md).

## Update and roll back

Commit the change, then build and deploy:

```bash
make image
```

```bash
make deploy
```

The new tag carries the new commit, and the deploy starts a new release. Runpod replaces the
workers gradually, and `flux-endpoint status` marks the old ones as stale. To roll back,
`make rollback` finds the build before this one in the git history and deploys it:

```bash
make rollback
```

`make rollback` does not check the registry. If `make image` never ran for that commit, the tag
does not exist and the workers fail to pull it, so check the tag on Docker Hub first. To deploy
any earlier build, run `make deploy` with `TAG` set to that build's tag.

## Variants

The model can also come from a Runpod cached model or a Global Volume, and both use the public
`slim` image. For a cached model in production, use `deploy/cached-model.toml`. The specs in
`deploy/cold-start/` are the study endpoints behind the comparison in [Results](results.md).

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

Then set the **Model** and **Max workers** in the console, in one edit, as in step 2 of [way
3](try-it.md#3-deploy-in-your-own-runpod-account). Otherwise, a worker that Runpod prepared
before the model was set can restart in a loop while it bills. Runpod downloads the model to
hosts before workers start and does not bill for the download. Once the model is set, change
`workers_max` in the spec to 2, or a later deploy sets it back to 0.

### Global Volume

A Global Volume holds one copy of the weights for every data center, mounted at
`/runpod-volume` on every worker. For this model it is the slowest option, at about 6 minutes
per cold start (see [Results](results.md#where-the-model-comes-from)).

For the study, I filled one like this:

1. Create the volume with `POST /v2/global-volumes` and a name. Global Volumes are in beta.
2. Start a CPU Pod from the image with the weights, with the registry credential and the
   volume mounted at `/data`. Give it the command
   `sh -c 'cp -r --no-preserve=all /models/flux-1-dev /data/ && sleep infinity'`. `cp -a`
   fails there, because the volume cannot store file timestamps. In my run, the copy plus a
   byte-by-byte compare of every file took about 5.5 minutes.
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

- Billing is per second, from the worker start through the model load and the jobs to the end
  of the idle timeout. Image pulls and cached model downloads normally happen while the worker
  is scaled down, and Runpod does not bill them. If a job starts a worker before the image is
  on the host, the pull is billed as part of the cold start.
- The largest hourly spend is `workers_max` times the GPU price. With 2 workers it is $3.50 per
  hour on L40S, and $5.44 if every worker falls back to an A100.
- Every job has a 600 second limit, and every job from the CLI has a time to live, so a stuck
  or abandoned job stops billing.
- With no active workers in the spec, an idle endpoint costs nothing.
- Runpod's default account limit is $80 of spend per hour.
