"""
Simple, lightweight progress monitor for whichever Chinchilla-sweep model
is CURRENTLY training. Single-line, in-place update (like pip/tqdm-style
download bars), fast refresh, moving-average ETA.

Performance note: this only reads the TAIL of the training CSV (via `tail`,
not a full re-parse with pandas) on every check -- cheap regardless of how
large the log has grown, which is what makes a short refresh interval safe.
A naive full-file re-parse every few seconds would add real, growing disk/
CPU cost competing with the actual training process; this avoids that.

Run:
    python3 dev_scripts/monitor_current_model.py
    python3 dev_scripts/monitor_current_model.py --interval 3 --window 10
"""
import os, glob, subprocess, argparse, sys, time
from collections import deque

CKPT_ROOT = "/workspace/PP_collision/checkpoints"
RUN_NUM = "sweep"
TAIL_LINES = 100  # how many lines of the CSV to read each check -- cheap regardless of total file size
PLOT_SCRIPT = "/workspace/PP_collision/dev_scripts/plot_old_vs_chinchilla.py"
PLOT_ERROR_LOG = "/tmp/chinchilla_plot_errors.log"
PLOT_INTERVAL = 300.0  # force a replot at least this often (seconds), even with no new completions

CONFIGS = {
    "scale_w64_d12_chinchilla":   100_000,
    "scale_w128_d12_chinchilla":  100_000,
    "scale_w256_d12_chinchilla":  100_000,
    "scale_w512_d12_chinchilla":  200_000,
    "scale_w1024_d12_chinchilla": 800_000,
    "scale_w1536_d12_chinchilla": 1_600_000,
}


def find_sweep_dir():
    candidates = sorted(glob.glob(f"{CKPT_ROOT}/Chinchilla_Sweep_*"))
    if not candidates:
        raise SystemExit("No Chinchilla_Sweep_* folder found.")
    return candidates[-1]


def find_active_config(sweep_dir):
    # [FIX] was: `ps`-based detection -- unreliable across container
    # boundaries. Docker containers have isolated process namespaces, so a
    # monitor running in a DIFFERENT container instance than training
    # (e.g. a fresh `ppcontainer` invocation in a new terminal) genuinely
    # cannot see the training process via `ps`, even as root, even on the
    # same machine. The CSV log files, however, live on the shared
    # /workspace/PP_collision filesystem and are visible from any
    # container -- whichever config's CSV was modified most recently is
    # the one actively being written to right now.
    best_cfg, best_mtime = None, 0
    for cfg_name in CONFIGS:
        csv_path = f"{sweep_dir}/{cfg_name}/{RUN_NUM}/config_{cfg_name}_run_{RUN_NUM}.csv"
        if not os.path.exists(csv_path):
            continue
        mtime = os.path.getmtime(csv_path)
        if mtime > best_mtime:
            best_mtime, best_cfg = mtime, cfg_name
    # sanity check: only call it "active" if written to recently (within
    # the last couple minutes) -- otherwise every config is finished/idle
    # and nothing is currently training.
    if best_cfg is not None and (time.time() - best_mtime) < 120:
        return best_cfg
    return None


def tail_last_train_step(sweep_dir, cfg_name):
    """Cheap: reads only the last TAIL_LINES lines of the CSV via `tail`,
    not the whole file. Returns the most recent train-split step, or None."""
    csv_path = f"{sweep_dir}/{cfg_name}/{RUN_NUM}/config_{cfg_name}_run_{RUN_NUM}.csv"
    if not os.path.exists(csv_path):
        return None
    try:
        out = subprocess.run(["tail", "-n", str(TAIL_LINES), csv_path],
                            capture_output=True, text=True).stdout
    except Exception:
        return None
    last_step = None
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        if parts[0].strip() != "train":
            continue
        try:
            last_step = int(float(parts[1]))
        except ValueError:
            continue
    return last_step


def bar(frac, width=30, color="\033[92m", bg="\033[100m"):
    # Solid Unicode block characters, eighth-increment partial blocks
    # (" ▏▎▍▌▋▊▉█") for smooth sub-character granularity. The unfilled
    # region uses plain spaces (not a glyph like ░) with a background
    # color -- a space is the most universally-rendered character there
    # is, so there's no glyph-consistency risk, and the background color
    # still fills the whole cell solid with no visible glyph needed.
    RESET = "\033[0m"
    frac = max(0.0, min(frac, 1.0))
    total_eighths = round(frac * width * 8)
    full_blocks = total_eighths // 8
    remainder_eighths = total_eighths % 8
    partial_chars = " ▏▎▍▌▋▊▉█"

    if full_blocks >= width:
        filled_str = f"{color}" + "█" * width + RESET
        rest_str = ""
    elif remainder_eighths == 0:
        filled_str = f"{color}" + "█" * full_blocks + RESET
        rest_str = f"{bg}" + " " * (width - full_blocks) + RESET
    else:
        filled_str = f"{color}" + "█" * full_blocks + RESET
        partial_char = partial_chars[remainder_eighths]
        empty_count = width - full_blocks - 1
        rest_str = f"{bg}{color}" + partial_char + " " * empty_count + RESET

    return f"{filled_str}{rest_str} {frac*100:5.1f}%"


def get_done_set(sweep_dir):
    """Cheap: reuses the same tail-based read as the progress bar itself.
    Checks every config, not just the currently active one."""
    done = set()
    for cfg_name, total in CONFIGS.items():
        step = tail_last_train_step(sweep_dir, cfg_name)
        if step is not None and step >= total:
            done.add(cfg_name)
    return done


def run_plot():
    """Runs plot_old_vs_chinchilla.py as a subprocess, completely silently
    -- no output at all, so the live progress bar is never interrupted,
    whether this fires from the 5-minute cadence or a model just
    finishing. Errors are logged to a file (not stdout) so problems can
    still be diagnosed without ever touching the terminal."""
    try:
        result = subprocess.run(["python3", PLOT_SCRIPT], capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            with open(PLOT_ERROR_LOG, "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] plot script failed:\n{result.stderr}\n\n")
    except subprocess.TimeoutExpired:
        with open(PLOT_ERROR_LOG, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] plot script timed out after 300s\n\n")
    except Exception as e:
        with open(PLOT_ERROR_LOG, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] plot script errored: {e}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=5.0,
                        help="seconds between checks (default: 5 -- safe due to tail-only reads)")
    parser.add_argument("--window", type=int, default=12,
                        help="number of recent readings to average over for the ETA (default: 12)")
    parser.add_argument("--plot-interval", type=float, default=PLOT_INTERVAL,
                        help=f"seconds between forced plot refreshes even with no new completions (default: {PLOT_INTERVAL:.0f} = 5min)")
    parser.add_argument("--no-plot", action="store_true",
                        help="disable auto-replotting entirely, just show the progress bar")
    args = parser.parse_args()

    sweep_dir = find_sweep_dir()
    history = deque(maxlen=args.window)  # (timestamp, step)

    prev_done = set()
    last_plot_time = 0.0
    if not args.no_plot:
        prev_done = get_done_set(sweep_dir)
        run_plot()  # initial plot on startup
        last_plot_time = time.time()

    try:
        while True:
            cfg = find_active_config(sweep_dir)
            if cfg is None:
                line = "no Chinchilla training process currently running"
            else:
                total = CONFIGS[cfg]
                step = tail_last_train_step(sweep_dir, cfg)
                if step is None:
                    line = f"{cfg}: waiting for first log entry..."
                else:
                    now = time.time()
                    history.append((now, step))

                    eta_str = "gathering rate data..."
                    if len(history) >= 2:
                        t0, s0 = history[0]
                        t1, s1 = history[-1]
                        dt, dstep = t1 - t0, s1 - s0
                        if dt > 0 and dstep > 0:
                            rate = dstep / dt  # steps/sec, moving average over the window
                            remaining = total - step
                            eta_hours = (remaining / rate) / 3600
                            eta_str = f"ETA {eta_hours:.2f}h  ({rate:.2f} steps/sec, {len(history)}-sample avg)"

                    frac = step / total
                    line = f"{cfg}: {bar(frac)}  {step:,}/{total:,}  {eta_str}"

            # single-line in-place update: \r returns cursor to line start
            # WITHOUT a newline, so the next write overwrites it in place --
            # the classic pip/tqdm progress-bar technique. Pad with spaces
            # to fully overwrite anything longer from the previous line.
            sys.stdout.write("\r" + line.ljust(120))
            sys.stdout.flush()

            if not args.no_plot:
                now_done = get_done_set(sweep_dir)
                newly_done = now_done - prev_done
                if newly_done:
                    # [FIX] was: sys.stdout.write(f"\n[...] Newly completed: ...")
                    # -- that explicit newline broke the seamless \r-based
                    # bar update right at a model transition, which is
                    # exactly the moment it matters most. run_plot() is
                    # already fully silent; the bar itself already reflects
                    # the new active model on the next iteration naturally.
                    run_plot()
                    last_plot_time = time.time()
                    prev_done = now_done
                elif (time.time() - last_plot_time) >= args.plot_interval:
                    run_plot()
                    last_plot_time = time.time()

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()