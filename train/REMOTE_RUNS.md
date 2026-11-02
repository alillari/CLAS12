# Run CLAS12 jobs on an edgexpert

Use this after a successful [single adapter run](TRAINING_RECIPES.md). The
training commands and scientific choices stay the same; this guide explains
where they execute, how to keep them running after disconnecting, and how to
recover their results. It also applies to the [worked example](WORKED_EXAMPLE.md),
[campaigns](downstream/campaign/GETTING_STARTED.md) and
[tuning](downstream/tuning/GETTING_STARTED.md).

## Know which machine and path you are using

| Location | Role |
| --- | --- |
| Laptop checkout | Edit/review source and prepare a specific source revision. |
| Remote host reached with SSH | Owns the GPU, Docker daemon, input datasets and persistent output directories. |
| Container on that host | Runs Python with the selected CUDA/Mamba environment. Paths in its YAMLs must exist **inside this container**. |
| Laptop SSHFS view of remote files | Useful for inspection. Running Python in that directory still runs Python on the laptop. |

Get an SSH alias/account, Docker access, an approved image and the dataset paths
from whoever manages the machine. `alessio-edge` (a02c) and `alessio-aux` (a518)
are existing operator aliases, not accounts automatically available to new users.
Use your own configured alias. On the **laptop**:

```bash
export EDGE_HOST=your-edgexpert-ssh-alias
ssh -o ClearAllForwardings=yes "$EDGE_HOST"
```

`ClearAllForwardings` avoids starting unrelated port forwards from your SSH
configuration. If connection fails, distinguish hostname resolution, VPN/host
reachability and SSH authentication before changing a job. A lost connection
alone says nothing about whether a detached job is still running.

## 1. Inspect the host and reserve a new run directory

The following blocks run on the **remote host**, unless marked laptop:

```bash
hostname
uname -m
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
nvidia-smi

export EDGE_JOB=clas12-momentum-onboarding-001
export EDGE_RUN=/absolute/remote/results/remote_runs/$EDGE_JOB
export EDGE_TRAIN_DATA=/absolute/remote/training_product
export EDGE_VAL_DATA=/absolute/remote/validation_product
export EDGE_IMAGE=pp_collision:alessio-latest
mkdir -p "$(dirname "$EDGE_RUN")"
mkdir "$EDGE_RUN"
mkdir "$EDGE_RUN/source" "$EDGE_RUN/config" "$EDGE_RUN/results"

export EDGE_IMAGE_ID="$(docker image inspect "$EDGE_IMAGE" --format '{{.Id}}')"
docker image inspect "$EDGE_IMAGE_ID" > "$EDGE_RUN/image.json"
```

Choose a unique job/directory name; `mkdir` fails if it already exists. Inspect
existing workloads before allocating another GPU job. The image tag above was
used in prior team runs; inspect the image on the selected host and record its
immutable ID. Image availability on one host does not establish availability on
the other. GB10 hosts use ARM64 images and compiled kernels appropriate to that
GPU; an x86 laptop environment is not an interchangeable runtime.

This step records an existing image; it does not build or replace an environment.
The dated [Mamba runtime record](MAMBA_SUPPORT.md#mamba3-runtime) describes prior
validation, not a guarantee for every source/image combination.

## 2. Transfer an exact source snapshot

On the **laptop**, from the intended repository checkout, set the same remote
run path and copy a committed snapshot:

```bash
export EDGE_RUN=/absolute/remote/results/remote_runs/clas12-momentum-onboarding-001
export EDGE_STAGE="$(mktemp -d)"
git status --short
git rev-parse HEAD > "$EDGE_STAGE/source.commit"
git archive --format=tar --output="$EDGE_STAGE/source.tar" HEAD
scp -o ClearAllForwardings=yes "$EDGE_STAGE/source.tar" "$EDGE_STAGE/source.commit" "$EDGE_HOST:$EDGE_RUN/"
```

`git archive HEAD` includes committed files only. If the intended change is
uncommitted, save it on your working branch first or use a separately recorded
working-tree snapshot. Do not assume a commit ID describes dirty source.

Back on the **remote host**, with the step-1 variables still set:

```bash
tar -xf "$EDGE_RUN/source.tar" -C "$EDGE_RUN/source"
sha256sum "$EDGE_RUN/source.tar" > "$EDGE_RUN/source.tar.sha256"
cp "$EDGE_RUN/source/scripts/configs/clas12_adapter_recipes.yaml" "$EDGE_RUN/config/model.yaml"
```

This leaves the host's primary checkout and other studies untouched. A source
snapshot, copied model/analysis recipes, image ID and input identifiers together
identify the run; a branch name alone can move while the job is running.

## 3. Give the job container paths and persistent outputs

This example runs the existing 50k-track AdapterOnly momentum recipe. It creates
its stats inside the container from the same training cohort, then fits. It is
a full 30k-update recipe, **not a short GPU smoke test**. Change budgets and
filters deliberately in the copied model and corresponding commands if needed.

| Host path | Container path | Access |
| --- | --- | --- |
| `$EDGE_RUN/source` | `/workspace` | Read-only code snapshot. |
| `$EDGE_RUN/config` | `/config` | Read-only commands and copied recipes. |
| `$EDGE_TRAIN_DATA` | `/inputs/train` | Read-only training product. |
| `$EDGE_VAL_DATA` | `/inputs/validation` | Read-only validation product. |
| `$EDGE_RUN/results` | `/results` | Writable logs, stats, checkpoints, plots and caches. |

On the remote host, write the job's public commands:

```bash
cat > "$EDGE_RUN/config/command.sh" <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
python train/downstream/scripts/compute_regression_target_stats.py \
  --data-root "$CLAS12_DATA_ROOT" --split pretrain --task mom \
  --adapter-sample-mode event_segment --segment-target-source mctrue \
  --segment-min-clusters 12 --low-thr 1 --high-thr 100 --limit-size 50000 \
  --output "$CLAS12_MOMENTUM_STATS"
python train/downstream/train_track_regression.py \
  --yaml_config /config/model.yaml --config clas12_momentum_50k \
  --run_num momentum_seed42 --eventnumber 50000 --train_batch_size 128 --seed 42 \
  --checkpoint_dir /results/momentum/checkpoints \
  --log_file_name training.log --checkpoint_file_name adapter.pth \
  --resolved_config_path /results/momentum/resolved.json \
  --artifact_summary /results/momentum/artifacts.json
JOB

cat > "$EDGE_RUN/config/launch.sh" <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
finish() {
  rc=$?
  trap - EXIT
  printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > /results/launcher_result.txt
  exit "$rc"
}
trap finish EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace
export MPLBACKEND=Agg MPLCONFIGDIR=/results/cache/matplotlib
export XDG_CACHE_HOME=/results/cache TORCH_EXTENSIONS_DIR=/results/cache/torch_extensions
export TRITON_CACHE_DIR=/results/cache/triton
export CLAS12_ARTIFACT_ROOT=/results
export CLAS12_DATA_ROOT=/inputs/train CLAS12_VALIDATION_DATA_ROOT=/inputs/validation
export CLAS12_MOMENTUM_STATS=/results/stats/momentum_stats_train50k.json
mkdir -p /results/stats /results/cache
printf 'started_utc=%s\n' "$(date -u +%FT%TZ)" > /results/launcher_started.txt
cd /workspace
bash /config/command.sh > /results/launch.log 2>&1
LAUNCH
```

For finding, pretraining, a campaign or a study, replace `command.sh` with the
checked commands from that workflow and add the required mounts/environment.
Keep pretrained weights accessible read-only and point YAMLs at their container
path. Evaluation needs its final product, auxiliary metadata and original
training trees too. Copy/render campaign manifests **for the container paths**;
a laptop or host absolute path is not automatically translated by Docker.

Use `/results` for every durable output. Read-only source also means bytecode,
Matplotlib, Triton/extension and optional W&B caches need writable destinations.
The example redirects common caches and disables source bytecode writes. If
using W&B, select its mode explicitly and place its files under `/results` too.

## 4. Launch detached and reconnect later

Still on the **remote host**:

```bash
docker run -d --name "$EDGE_JOB" --restart=no --init --network none \
  --gpus all --shm-size 2g --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --workdir /workspace \
  --mount "type=bind,source=$EDGE_RUN/source,target=/workspace,readonly" \
  --mount "type=bind,source=$EDGE_RUN/config,target=/config,readonly" \
  --mount "type=bind,source=$EDGE_TRAIN_DATA,target=/inputs/train,readonly" \
  --mount "type=bind,source=$EDGE_VAL_DATA,target=/inputs/validation,readonly" \
  --mount "type=bind,source=$EDGE_RUN/results,target=/results" \
  "$EDGE_IMAGE_ID" bash /config/launch.sh > "$EDGE_RUN/container.id"
docker inspect "$EDGE_JOB" > "$EDGE_RUN/container.json"
```

Docker's daemon now owns the container, so closing SSH or the laptop does not
end the job. No `tmux` session is required. `--restart=no` does not relaunch it
after host/daemon restart; the run also has no automatic optimizer resume.
The example retains the stopped container for inspection instead of using
`--rm`. Outputs persist in the bind-mounted run directory either way.

`--user` makes outputs belong to the launching account. Confirm that account
can read the inputs and write the results. `--shm-size 2g` is a starting shared
memory allocation for worker IPC, not a CPU/GPU memory limit. Increase it if
measured loader needs justify it. `--gpus all` exposes available GPUs; it does
not reserve an idle GPU or coordinate with other jobs. On a multi-GPU host,
select the agreed device explicitly. CPU/RAM/worker allocations also need to
fit the host workload. `--network none` fits local data/offline logging; online
W&B requires an appropriate network setting and account configuration.

## 5. Monitor processes, progress and completion

After reconnecting, restore `EDGE_JOB` and `EDGE_RUN` in the remote shell:

```bash
docker ps -a --filter "name=$EDGE_JOB"
docker inspect "$EDGE_JOB" --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} error={{.State.Error}}'
docker top "$EDGE_JOB" -eo pid,etime,pcpu,pmem,args
tail -n 40 "$EDGE_RUN/results/launch.log"
tail -n 10 "$EDGE_RUN/results/momentum/checkpoints/training.log"
docker stats --no-stream "$EDGE_JOB"
nvidia-smi
```

`docker top` requires a running container. The example sends Python output to
`launch.log`, so `docker logs` need not contain the training log. For campaigns,
also inspect their `status.yaml`, active row log and receipt; higher-level
historical launchers may have their own `status.json` or `completed.json`.
Those extra files are launcher-specific, not a universal campaign API.

A running container can be preparing data, evaluating, plotting or stuck. Check
its actual processes, recent step-log updates and timestamps together. Sample
GPU activity over time before calling it idle. Start with one runner/GPU;
additional concurrency should improve total completed work per wall-clock hour,
including evaluation, rather than merely increase instantaneous utilization.

After exit, reconcile all of:

- Docker exit code, OOM/error fields and `launcher_result.txt`.
- Expected checkpoint and training `artifacts.json`, including actual final step
  and checkpoint-selection result; load/evaluate the saved weights.
- Evaluation `summary.json`, actual accepted populations, finite predictions,
  metric definitions and expected plots when evaluation is part of the command.

The example stops after fitting. Add the matching evaluation/plot commands from
[the worked example](WORKED_EXAMPLE.md#5-evaluate-on-the-reserved-final-population)
when those stages are required. A zero shell exit plus an existing checkpoint
alone does not establish the intended scientific result. Shell EXIT traps may
not run after SIGKILL/OOM/host loss; a missing receipt requires inspection.

## 6. Retry the failed stage and retrieve artifacts

Read the first relevant error and preserve failed-attempt logs. Correct a path,
metadata, validation cap or environment issue before reusing GPU time. Use a
new run directory/container name for a changed recipe or fresh standalone fit.
An existing name after an SSH timeout may represent a successfully launched
job: inspect it before issuing another launch.

For campaign retries, follow the [restart controls](downstream/campaign/GETTING_STARTED.md#6-restart-select-rows-or-repeat-evaluation).
In particular, regression supports `--skip-train --force-eval` for retrying
inference with saved weights. Regenerate plots from saved results when possible.
Restarting an interrupted adapter trainer starts a fresh fit; pretraining has
its own checkpoint-resume behavior. Neither implies exact data-cursor resume.

On the **laptop**, copy the outputs and the run specification together:

```bash
export EDGE_LOCAL_RESULTS=/absolute/local/results/clas12-momentum-onboarding-001
mkdir -p "$EDGE_LOCAL_RESULTS"
rsync -av -e 'ssh -o ClearAllForwardings=yes' "$EDGE_HOST:$EDGE_RUN/" "$EDGE_LOCAL_RESULTS/"
```

Retain the source revision/archive, image record, model/analysis configs,
normalization stats, launcher outcome, receipts, logs and weights. Copied YAMLs
and receipts may retain container paths: remap them explicitly for local replay
while preserving the original record. Avoid broad Docker prune or dataset
cleanup as part of fetching or retrying a job.

## Before a long job: make a bounded execution check

Earlier real-data runs exposed preparation errors that import checks could not:
missing subset `metadata.json`, misaligned sidecars and a validation batch cap
smaller than the requested sample. Use these lessons for a short integration run:

1. Copy an aligned small event cohort with every consumed sidecar and metadata.
   Record original event IDs; do not copy arbitrary folders independently.
2. Generate momentum stats from its accepted training tracks. Keep the cohort,
   filters and event/track sample units explicit.
3. Bound updates **and** validation size/cadence in a copied model profile.
   With inherited loaders that always set `prefetch_factor`, use at least one
   training data worker; zero workers can fail before the first batch.
4. Exercise the actual GPU forward/backward kernel, save a checkpoint, reload
   it with matching architecture and run the standalone evaluator.
5. Inspect counts and finite values as well as subprocess exits. Preserve
   failed attempts and the final corrected result separately.

[CUDA tests](MAMBA_SUPPORT.md#checkpoints-and-validation) complement this check.
A skipped CUDA test or successful import does not validate the kernels. A few
optimizer updates establish execution, not convergence or useful physics quality.

## If an assistant operates the jobs

Provide a job brief containing host, exact source revision, image, input roots,
cohort/filter rules, backbone/checkpoint, label and update budgets, seeds,
validation/final populations, allowed resources and required output artifacts.
Require a returned job/container ID, persistent output path and exact launch
plan so another person can monitor or reproduce it.

Where the separately installed `clas12-edge` integration is available, its
workflow is `preview_run` → `prepare_run` → `launch_run`, followed by
`get_run_status`, `read_run_logs` and artifact inspection. It snapshots source
read-only at `/workspace`, writes persistent outputs at `/results` and requires
explicit mounts for inputs under its configured project root. Its local
`source_path` captures tracked modifications and nonignored new files, unlike
`git archive HEAD`; preserve the resulting source checksum record. Reuse the **same job ID** to query/retry after a
transport timeout; changing the plan needs a new ID. Its `run_tests` convenience
path is for bounded probes; inspect the selected runtime deadline and GPU flag.
The integration's successful state means exit 0 plus declared artifacts, so the
scientific checks above still apply. Its client installation/configuration is
separate from this repository's training commands; plain SSH/Docker remains usable.
