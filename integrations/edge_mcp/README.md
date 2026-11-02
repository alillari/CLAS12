# CLAS12 Edge MCP

Local MCP tools and a matching CLI for **a02c** (`alessio-edge`) and **a518**
(`alessio-aux`). Reuses the existing SSH configuration, keys, Docker daemon,
training image and data paths. Requires no remote MCP service, network listener,
host package installation, or changes to either primary source checkout.

Version **0.2.0** adds direct remote file/image inspection, one-call code tests,
and detached Docker environment builds. The server exposes 19 tools. Existing
job IDs, plans and running containers remain usable.

## Installed use

The isolated environment is `.venv/` in this directory. The registered server is
`clas12-edge`; a new Codex chat/client refresh may be needed to load newly added
tools. The tools are tested through a real MCP STDIO client, independently of
whether the current chat has refreshed its tool inventory.

```bash
edge_root=/home/alessio/ML-work/CLAS12/integrations/edge_mcp
"$edge_root/.venv/bin/edge-runner" hosts
"$edge_root/.venv/bin/edge-runner" inspect a02c
"$edge_root/.venv/bin/edge-runner" inspect a518 --fast
"$edge_root/.venv/bin/edge-runner" list a02c
```

Reinstall the tested Python 3.11 environment with `./install.sh`. Use
`./install.sh --register` to also register the MCP server in Codex. Other
Python versions need compatible wheels and should be validated separately.
The environment is independent of the ML conda environments. `requirements.lock`
records the tested dependency versions; no OpenAI API key is needed.

## Tools

| Tool | Effect |
| --- | --- |
| `list_hosts` | List configured aliases without connecting |
| `inspect_host` | Read host resources and optionally active containers/processes |
| `preview_run` | Resolve image, source checksums and mount paths; no remote writes |
| `prepare_run` | Save an immutable source snapshot and execution plan; no launch |
| `launch_run` | Create/start a detached container, or return its existing identity |
| `get_run_status` | Reconcile Docker state, supervisor result and declared artifacts |
| `list_runs` | List recent registered jobs, including completed jobs |
| `read_run_logs` | Read bounded persisted stdout or Docker logs |
| `read_run_artifact` | Read a small UTF-8 output such as JSON metrics |
| `stop_run` | Stop only the matching registered container; preserve its outputs |
| `list_remote_files` | List/paginate files directly over SSH |
| `read_remote_file` | Read/paginate a remote configuration, source or log file |
| `search_remote_files` | Find literal text with filenames and line numbers |
| `list_images` | List local Docker image tags and IDs |
| `inspect_image` | Inspect architecture, user, entrypoint and image identity |
| `run_tests` | Stage current local/remote code, launch a probe and return logs/status |
| `build_environment` | Start a derived Docker image build with a saved context |
| `get_environment_build` | Read build state, logs and resulting image ID |
| `stop_environment_build` | Cancel only a matching managed build worker |

MCP tools have structured input/output schemas and read/write annotations.
Approval policy remains the client's configuration; these annotations do not
provide an operating-system authorization boundary. The SSH account retains
its existing Docker privileges.

## Exploratory workflow

Ask Codex to inspect the relevant remote source/configuration, edit code locally,
test it on the selected host, and build an isolated image if dependencies need
to change. These tools do not require a finished experiment recipe. For example:

> Read the historical pretraining config on a02c, test this local change on a518,
> then build a separate environment with the pinned dependencies and rerun it.

### Direct file and image inspection

The file tools read directly through SSH; there is no diagnostic container,
source staging, remote script file, or remote write. Paths are absolute and
resolve below `/home/alessio/ML-work`, covering the source trees, datasets and
study/job artifacts. Symlinks escaping that root are rejected. Directory scans
do not follow symlink directories or traverse `.git`.

```bash
edge_root=/home/alessio/ML-work/CLAS12/integrations/edge_mcp
"$edge_root/.venv/bin/edge-runner" files a02c /home/alessio/ML-work/CLAS12/scripts/configs
"$edge_root/.venv/bin/edge-runner" read a02c /home/alessio/ML-work/CLAS12/requirements.txt
"$edge_root/.venv/bin/edge-runner" search a02c /home/alessio/ML-work/CLAS12/fm4npp \
  knn_later_indices_batch --pattern '*.py'
"$edge_root/.venv/bin/edge-runner" images a518
"$edge_root/.venv/bin/edge-runner" image a518 pp_collision:alessio-latest
```

Reads support byte `offset`, `max_bytes` (up to 64 KiB), and `next_offset`.
Binary files are rejected; use a format-aware Python probe through `run_tests`
to inspect arrays/checkpoints. Listings support glob `pattern`, recursion,
hidden entries and offset pagination. Searches are literal and case-sensitive,
with path/line matches; patterns filter relative filenames. Scan/time/output
budgets prevent large dataset directories from overwhelming a call.
`truncated` or `scan_truncated` means refine the search or continue pagination;
an incomplete search cannot establish that a match is absent.

`inspect_image` exposes image identity, architecture, user and entrypoint plus
environment variable **names**, without returning their values. Installed
package versions can be inspected with the package-probe example below.

### Edit and test

`run_tests(host, spec, source_path=..., wait_seconds=10)` combines preparation,
launch, a short wait and bounded stdout retrieval. `spec` is a `TestSpec`, with
the same fields as `RunSpec`. Use an argv command such as `python -m unittest ...`
or `python -c ...`, with a stable unique `job_id` for that revision/test attempt.

Local `source_path` captures working-tree changes and nonignored new files. Use
`remote_source_path` instead to snapshot a remote checkout/context, including
its current changes. Neither source tree is edited. Source is read-only at
`/workspace`; scratch and test reports can go to `/tmp` or `/results`.
Omit both paths for inline probes that need no source files.

Tests default to CPU execution with a **300-second optional deadline**; set
`gpu=true` for CUDA tests, change the deadline, or set it to `null`. CPU/RAM/PID
caps remain optional. These test defaults do not change `RunSpec` training
defaults, which remain uncapped with no runtime deadline.

```bash
edge_root=/home/alessio/ML-work/CLAS12/integrations/edge_mcp
"$edge_root/.venv/bin/edge-runner" test a518 \
  --spec "$edge_root/examples/package-probe.json" --wait 10
```

For source tests add `--source /absolute/local/checkout`, or
`--remote-source /home/alessio/ML-work/REMOTE-CHECKOUT`. After an intentional
code change, choose a new job ID. Repeating the same ID/plan reuses its container.
Only explicit lock-busy responses are retried automatically for up to five
seconds, with the same ID. Transport timeouts are not blindly retried.

`wait_seconds` is between 0 and 30; longer tests continue detached and return a
job ID. Inspect them with the existing status/log/stop tools. A failed test
returns `status=failed`, its exit code and stdout, so a development loop can
read the failure, edit code and submit the next revision. Source/context
snapshots retain the existing 64 MiB/20,000-file bounds; mount large datasets
explicitly instead of including them in the source archive.

### Build and test a dependency environment

`build_environment` accepts an `environment_id`, a local `base_image`, and
Dockerfile **instructions after FROM**. For example, supply a `RUN python -m
pip install ...` instruction with the desired pinned package versions.
Use local `source_path` or `remote_context_path` for files needed by `COPY`.
The tool resolves the base image ID, creates a content-addressed local base tag,
and generates FROM. Extra FROM/ONBUILD stages are outside this tool's supported
Dockerfile format. The base image must already exist on the selected host.

```bash
edge_root=/home/alessio/ML-work/CLAS12/integrations/edge_mcp
"$edge_root/.venv/bin/edge-runner" build-env a518 exploratory-env-example \
  --instructions "$edge_root/examples/environment.Dockerfile" --network none
"$edge_root/.venv/bin/edge-runner" env-status a518 exploratory-env-example
```

Build output is tagged only as `clas12-explore/ENVIRONMENT-ID:local`. Existing
tags are never intentionally overwritten, and nothing is pushed to a registry.
The build installs dependencies **inside the new image**. It does not install
host packages, change the primary checkout, replace the training image, or
modify containers already running from that image. The result is checked against
the expected labels, architecture and base filesystem layers.

The call returns promptly. A detached, standard-library host worker runs Docker
with the default local builder; no build daemon/service is installed by this
integration. Context, Dockerfile, base ID, fingerprint, worker identity and logs
are saved under `edge_mcp/environments/ENVIRONMENT-ID/`. The final image ID from
`get_environment_build` can be passed directly as `spec.image` to `run_tests`
or a subsequent training preparation.

Builds have no wall-clock deadline. `network=default` permits RUN instructions
to fetch packages; `network=none` isolates their network access. Build arguments,
secret forwarding, multi-stage Dockerfiles and remote registry publishing are
not exposed by this first build tool. Builds can consume substantial CPU/disk;
inspect the host and choose when to run large compilations.

Repeat the same ID/plan to retrieve the existing build. A changed or deliberately
retried failed build needs a new environment ID. On connection loss, query the
original ID before taking another action. Cancellation uses
`stop_environment_build` / `env-stop`; it verifies the worker's process start
token and command, then uses a Linux process descriptor to avoid signalling a
reused PID. Cancellation is a request; query status for the final result. Logs,
contexts and built images are retained, with no automatic cleanup.

## Prepare and run

`examples/gpu-smoke.json` and `examples/smoke_job.py` provide a small real CUDA
forward/backward check. It reads no checkpoint contents or training data.
Choose a new `job_id` in a copy of the JSON for each intentional new execution.

```bash
edge_root=/home/alessio/ML-work/CLAS12/integrations/edge_mcp
"$edge_root/.venv/bin/edge-runner" preview a02c \
  --spec "$edge_root/examples/gpu-smoke.json" --source "$edge_root/examples"
"$edge_root/.venv/bin/edge-runner" prepare a02c \
  --spec "$edge_root/examples/gpu-smoke.json" --source "$edge_root/examples"
"$edge_root/.venv/bin/edge-runner" launch a02c gpu-smoke-example
"$edge_root/.venv/bin/edge-runner" status a02c gpu-smoke-example
"$edge_root/.venv/bin/edge-runner" logs a02c gpu-smoke-example --tail 30
"$edge_root/.venv/bin/edge-runner" artifact a02c gpu-smoke-example gpu.json
```

The same operations work on a518. For an active registered job:

```bash
/home/alessio/ML-work/CLAS12/integrations/edge_mcp/.venv/bin/edge-runner \
  stop a518 YOUR-JOB-ID
```

### Execution contract

- Commands are argument arrays, without implicit shell expansion. A shell can be
  invoked explicitly as an executable if an experiment requires one.
- `source_path` is **local**. In a Git checkout, the snapshot includes the current
  contents of tracked files and nonignored untracked files; deletions are honored.
  It records the commit, dirty state, file checksums and executable modes. An empty
  source gives an empty workspace. Source symlinks are rejected. Snapshots are
  limited to 64 MiB and 20,000 files; preparation creates an isolated copy.
- The source snapshot is `/workspace` (read-only); this is the working directory.
  Persistent outputs belong in `/results`. Set `PYTHONPATH=/workspace` explicitly
  in `environment` if an experiment's invocation requires it.
- Extra mount sources are **remote** paths below `/home/alessio/ML-work`.
  Read-only mounts are the default. Additional writable mounts must be below
  `result_deep_storage`, cannot include the integration's records, and must not
  overlap another active managed job's writable mounts. Use narrow campaign
  directories. Legacy jobs launched outside this integration are observable but
  are not covered by its output locks; check host activity before launching.
- The image is resolved during preparation and launched by its exact local image
  ID with `--pull=never`. Both hosts currently share the tested ARM64 image.
  Preparation does not build, install into, or modify an image.
- Jobs run with the SSH user's numeric UID/GID. Source and runner mounts are
  read-only; `/results` remains owned by that user. The process has no added Linux
  capabilities and no new privileges.
- Defaults: GPU enabled, host IPC, bridge networking, and **no integration-imposed
  CPU quota, RAM cap, process cap or runtime deadline**. `cpus`, `memory_gib`,
  `pids_limit` and `timeout_seconds` default to `null`; only non-null values add
  corresponding restrictions. Host, Docker daemon and parent cgroup constraints
  still apply. Explicit deadlines have no fixed upper bound: four weeks is
  `2419200` seconds, and `null` permits longer runs without a scheduled kill.
  Optional limits are recorded in the immutable plan. The smoke examples/tests
  set small limits explicitly; those limits are not defaults for real training.
  Private IPC with 1 GiB shared memory and network isolation remain selectable.
  `RunSpec` contains all supported fields; unknown fields fail.
- `expected_artifacts` lists files relative to `/results`. Presence is checked;
  their scientific validity is not inferred. For an existing campaign, mount its
  configuration/data paths explicitly and retain its validation and checkpoint
  rules in the versioned experiment command.
- Changing a prepared plan requires a new job ID. Repeating preparation for the
  same source, runner, image and specification returns the original plan.

### Lifecycle and recovery

Version 0.1.1 removes the original 24-hour default deadline, seven-day maximum,
and default resource caps. Existing prepared/running jobs retain their saved
specification and supervisor; editing these defaults cannot change a running
container. Prepare a new job ID with the desired optional limits for future runs.

Every remote record is stored under:

```text
/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/JOB-ID/
  plan.json                 # host, resolved image, source identity, exact Docker argv and limits
  source_manifest.json      # per-file checksums and modes
  state.json                # launch/stop intent and timestamps, once launched
  source/                   # isolated source snapshot
  runner/                   # reusable supervisor and private execution specification
  output/stdout.log
  output/runner_result.json
  output/...                # command outputs
```

Job and host launch locks serialize integration mutations. The container has
labels tying it to the plan fingerprint. Launch uses create/start separately and
recovers an interrupted create/start using the same container. Repeating launch
never restarts a finished container. After an SSH timeout, inspect the **same job
ID** and retry that ID; do not create another job to resolve an ambiguous launch.

The Docker daemon owns detached execution. Closing Codex, its MCP subprocess,
SSH or the laptop does not stop a running job. Reconnect and use its job ID.
Containers are retained with restart policy `no`: host reboot survival/resume is
not automatic. The supervisor sends TERM then KILL to the command's process group
on timeout/stop; Docker also removes remaining container processes on exit.
Stop waits up to 15 seconds and does not delete results or prune Docker.

SSH and MCP request timeouts bound individual control calls; they do not impose a
training runtime deadline. `get_run_status` reports `effective_limits` from Docker
inspection together with the job's optional supervisor deadline.

Statuses include `prepared`, `created`, `running`, `succeeded`, `failed`, `stopped`,
`timed_out`, `incomplete` and `unknown` (plus Docker transient states).
`succeeded` requires Docker exit 0, a successful supervisor record, and every
declared artifact. `incomplete` covers a missing result/artifact after exit 0.
An unexpectedly removed container after launch is `unknown`, even if output files
exist; launch refuses to recreate it automatically. An interrupted launch with a
retained `created` container can be recovered using that same job ID.
Scientific validation is always reported as `not_assessed`.
Checkpoint continuation and skip-completed-arm behavior belong to the existing
training/campaign command, not the generic integration.

Stdout persists in the output directory; Docker logs rotate at 3 × 10 MiB.
Log reads are limited to 1,000 lines and 64 KiB. Text artifact reads are limited
to 64 KiB and cannot escape the output directory. Larger/binary artifacts can be
retrieved through normal SSH. There is intentionally no automatic cleanup.

## Verification

```bash
edge_root=/home/alessio/ML-work/CLAS12/integrations/edge_mcp
"$edge_root/.venv/bin/python" -m unittest discover -s "$edge_root/tests" -v
"$edge_root/.venv/bin/python" "$edge_root/tests/e2e.py" --hosts a02c a518 \
  --report "$edge_root/validation/e2e.json"
```

The live suite uses the actual SDK's STDIO client, discovers the tools and checks
structured output, then exercises monitoring, preview, preparation, CUDA
forward/backward, source staging, duplicate launch, logs/artifacts, MCP/SSH
disconnect/reconnect, stop with a TERM-ignoring descendant, failure, missing
artifact, deadline and retained job records. It creates only bounded synthetic
containers, stops its own active jobs on failure and leaves logs/records for audit.
It does not launch a production campaign. The test report is local and gitignored.
The local supervisor tests simulate elapsed times of three to five weeks; the
live suite also checks a short command with uncapped defaults through Docker
inspection, with a separate test-harness watchdog and cleanup.

The exploratory end-to-end suite is:

```bash
edge_root=/home/alessio/ML-work/CLAS12/integrations/edge_mcp
"$edge_root/.venv/bin/python" "$edge_root/tests/exploration_e2e.py" \
  --hosts a02c a518 --report "$edge_root/validation/exploration-e2e.json"
```

It builds a separate image with a tiny locally supplied Python package, observes
a failing test, edits the local test and verifies success in that image. It also
tests remote snapshots, direct file inspection, concurrent probes, reconnecting
to detached tests/builds, build failure/cancellation, and preservation of the
original image/primary source and the existing Mamba3 run. It uses CPU tests and
does not stop existing training. Test jobs/builds are stopped on failure; image
and record retention is intentional.

SSH requires existing VPN/DNS connectivity, trusted host keys and key access.
The transport preserves host-key verification, disables configured port forwards
and reuses private SSH sockets. This task's restricted shell sandbox blocked DNS
and the MCP subprocess handshake; live checks were run with authorized access
outside that sandbox. A configured MCP server must likewise have normal local
SSH/network access. It is not installed into the current model's tool inventory
merely by editing a config file.

The resource/process probe is a maintained copy of the existing Edge Monitor
`probe.py` from `/home/alessio/.local/share/edge-monitor`; the dashboard installation
is unchanged. The generic runner uses only the remote Python standard library.

Official references: [MCP configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli),
[building MCP tools](https://developers.openai.com/plugins/build/mcp-server),
and [Docker builds](https://docs.docker.com/reference/cli/docker/buildx/build/).
