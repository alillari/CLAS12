"""
Progress monitor for the Chinchilla sweep -- run this from a separate
terminal (e.g. VS Code) while the actual sweep runs in its own tmux
session. Shows a progress bar + ETA per model: DONE / actively training /
pending.

Rate (steps/sec) is measured empirically between successive runs of this
script via a small cached state file -- not guessed from file timestamps,
which can be unreliable. The first run after launching (or after this
script hasn't been run in a while) won't have an ETA yet; run it again a
minute or two later and the ETA will appear.

Run (from /workspace/PP_collision), re-run anytime to refresh:
    python3 dev_scripts/monitor_chinchilla_sweep.py

Auto-refresh every 30s:
    watch -n 30 python3 dev_scripts/monitor_chinchilla_sweep.py
"""
import os, glob, json, time, subprocess, argparse, sys, io
import pandas as pd

CKPT_ROOT = "/workspace/PP_collision/checkpoints"
RUN_NUM = "sweep"
STATE_FILE = "/tmp/chinchilla_monitor_state.json"

CONFIGS = [
    ("scale_w64_d12_chinchilla",   "m1", 100_000),
    ("scale_w128_d12_chinchilla",  "m2", 100_000),
    ("scale_w256_d12_chinchilla",  "m3", 100_000),
    ("scale_w512_d12_chinchilla",  "m4", 200_000),
    ("scale_w1024_d12_chinchilla", "m5", 800_000),
    ("scale_w1536_d12_chinchilla", "m6", 1_600_000),
]


def find_sweep_dir():
    candidates = sorted(glob.glob(f"{CKPT_ROOT}/Chinchilla_Sweep_*"))
    if not candidates:
        raise SystemExit("No Chinchilla_Sweep_* folder found under checkpoints/ yet.")
    return candidates[-1]  # most recent if somehow more than one


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


def get_last_step(sweep_dir, cfg_name):
    csv_path = f"{sweep_dir}/{cfg_name}/{RUN_NUM}/config_{cfg_name}_run_{RUN_NUM}.csv"
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, header=None, names=["split", "step", "loss", "lr"])
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        train = df[df["split"] == "train"].dropna(subset=["step"])
        if len(train) == 0:
            return 0
        return int(train["step"].max())
    except Exception:
        return None


def find_active_config(sweep_dir):
    # [FIX] was: `ps`-based detection -- fails across container boundaries
    # (Docker process namespaces are isolated; a monitor running in a
    # different container instance than training cannot see its process
    # via `ps`, even as root, even on the same machine). Use file mtime on
    # the shared filesystem instead -- whichever config's CSV was written
    # to most recently is the one actively training right now.
    best_cfg, best_mtime = None, 0
    for cfg_name, _, _ in CONFIGS:
        csv_path = f"{sweep_dir}/{cfg_name}/{RUN_NUM}/config_{cfg_name}_run_{RUN_NUM}.csv"
        if not os.path.exists(csv_path):
            continue
        mtime = os.path.getmtime(csv_path)
        if mtime > best_mtime:
            best_mtime, best_cfg = mtime, cfg_name
    if best_cfg is not None and (time.time() - best_mtime) < 120:
        return best_cfg
    return None


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def draw_once():
    sweep_dir = find_sweep_dir()
    active_cfg = find_active_config(sweep_dir)
    now = time.time()
    state = load_state()
    new_state = {}

    print(f"Chinchilla sweep progress -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"sweep dir: {sweep_dir}")
    if active_cfg:
        print(f"currently running process: {active_cfg}")
    else:
        print("no matching training process found right now (between runs, or check tmux)")
    print()

    for cfg_name, label, total_steps in CONFIGS:
        last_step = get_last_step(sweep_dir, cfg_name)

        if last_step is None:
            print(f"{label:>3} ({cfg_name}): PENDING (not started)\n")
            continue

        frac = last_step / total_steps
        is_done = frac >= 0.999
        is_active = (cfg_name == active_cfg)

        status = "DONE" if is_done else ("training..." if is_active else "idle / between runs")

        eta_str = ""
        prev = state.get(cfg_name)
        if not is_done and prev is not None:
            dt = now - prev["t"]
            dstep = last_step - prev["step"]
            if dt > 0 and dstep > 0:
                rate = dstep / dt  # steps/sec
                remaining = total_steps - last_step
                eta_hours = (remaining / rate) / 3600
                eta_str = f"  ETA: {eta_hours:.1f}h  ({rate:.2f} steps/sec)"
        elif not is_done:
            eta_str = "  ETA: (gathering rate data, re-run in a bit)"

        new_state[cfg_name] = {"step": last_step, "t": now}

        print(f"{label:>3} ({cfg_name}):")
        print(f"    {bar(frac)}  step {last_step:,}/{total_steps:,}  {status}{eta_str}\n")

    save_state(new_state)


def render():
    """Run draw_once() but capture its output as text, so we know exactly
    how many lines it produced before printing anything -- needed to move
    the cursor up the right amount on the NEXT refresh."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        draw_once()
    finally:
        sys.stdout = old_stdout
    text = buf.getvalue()
    return text, text.count("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true",
                        help="keep refreshing in place instead of drawing once and exiting")
    parser.add_argument("--interval", type=float, default=15.0,
                        help="seconds between refreshes when --loop is set (default: 15)")
    args = parser.parse_args()

    if not args.loop:
        draw_once()
        return

    prev_line_count = 0
    try:
        while True:
            text, n_lines = render()
            if prev_line_count:
                # [ELEGANT VERSION] instead of clearing the WHOLE screen:
                # move the cursor up exactly as many lines as last frame
                # printed (\033[{n}A), then clear only from there DOWN
                # (\033[J). Everything above the dashboard -- shell
                # history, earlier commands -- is left completely alone.
                sys.stdout.write(f"\033[{prev_line_count}A\033[J")
            sys.stdout.write(text)
            sys.stdout.write(f"\n(refreshing every {args.interval:.0f}s -- Ctrl+C to stop)\n")
            sys.stdout.flush()
            prev_line_count = n_lines + 1  # +1 for the "(refreshing...)" line
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()