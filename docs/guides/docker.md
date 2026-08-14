# Docker runtime

The Docker image is one reproducible runtime for local checks and the
Grid'5000 compute worker. It contains the locked `training` dependency extra,
but no project data, model files, checkpoints, Hugging Face token, SSH key, or
cache. The image targets `linux/amd64`, matching the existing Grid'5000 GPU
selection contract.

## Build and smoke test

Docker and the local `just` recipes are the only host requirements for these
checks:

```bash
just docker-build
just docker-smoke
```

The smoke test uses no network and no project-data mount. The default image
command is `grid5000_worker --help`; it does not audit, train, publish, or
submit a job. `UV_VERSION` and the Python image tag are explicit in
`Dockerfile`; its `linux/amd64` Python base image is pinned by digest. Debian
package versions are resolved from that base image's configured package mirror
at build time, so retain the built image digest as the deployment artifact.

## Local data contract

The application keeps its existing external data-root boundary. Mount that
root at the same absolute path inside the container:

```bash
IMAGE=osm-polygon-sentence-classifier:local
DATA_ROOT="/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier"

docker run --rm --platform linux/amd64 \
  --mount "type=bind,src=$DATA_ROOT,dst=$DATA_ROOT" \
  "$IMAGE" audit-landuse-dataset
```

The audit is the explicit review command and writes only its existing audit
artifacts below `.../audit/landuse`. Run it only when the dataset review is
authorized. An authorized training invocation is similarly explicit:

```bash
docker run --rm --platform linux/amd64 \
  --mount "type=bind,src=$DATA_ROOT,dst=$DATA_ROOT" \
  --env HF_TOKEN \
  "$IMAGE" python -c \
  'from osm_polygon_sentence_classifier.training import train_landuse_classifier; train_landuse_classifier()'
```

Training caches, checkpoints, model output, and Trackio state stay below the
mounted data root through the existing `ProjectConfig` and `ManagedPaths`
boundaries. Do not use a repository `data/`, `models/`, or cache directory as
an alternate mount.

Pass a token through `HF_TOKEN` or a read-only Hugging Face token-file mount;
never copy it into the image, Dockerfile, command history, or a durable data
artifact. The Grid'5000 worker maps the host token file read-only over the
writable per-run Hugging Face cache path, so model/data caches remain writable
without persisting the token in the run root. Publication and Trackio
synchronization remain opt-in training settings.

## Grid'5000 worker mode

The controller remains host-side. It performs SSH, checkout staging, live
usage-policy and quota checks, OAR scheduling, Hub provisioning, monitoring,
publication verification, and exact-root cleanup. The container performs only
the already-selected compute worker.

Provide an immutable, worker-local image reference and choose the runtime
explicitly when possible:

```bash
uv run grid5000-landuse run \
  --source-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --model-revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --container-image registry.example/osm-polygon-sentence-classifier@sha256:... \
  --container-runtime docker \
  --execute
```

The selected Grid'5000 node must already provide the requested Docker or
Podman runtime, access to the allocated GPU, and the exact image. The worker
mounts the clean host checkout read-only at `/home/app/checkout` and the
per-run persistent data root read-write at `/home/app/data`; the source code
and data are not copied into the image. It starts with a read-only container
filesystem, a non-root UID, and only a writable temporary filesystem plus the
explicit data mount. No `--privileged` mode or Docker-in-Docker is used.
The OAR-provided `CUDA_VISIBLE_DEVICES` value selects exactly one host GPU;
the container exposes that device as logical GPU `0`, which the worker then
checks with CUDA before training.

With `--container-runtime auto`, the worker checks Docker first and Podman
second, verifying that each daemon is accessible before selecting it. If the
runtime, image, mounts, data write, or GPU startup preflight is unavailable,
the worker exits with a clear failure and does not silently switch to the host
`uv` path. Omitting `--container-image` intentionally preserves the existing
non-container worker fallback. Container availability on remote Grid'5000
nodes cannot be verified from this local checkout; confirm it with the site
operator before an executing run.

The CLI rejects mutable image tags for remote workers. Build the image, publish
or preload it on the selected site, and pass its full `@sha256:<digest>`
reference so a continuation cannot silently use a different image.
