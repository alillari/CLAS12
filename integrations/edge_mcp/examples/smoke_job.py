"""Bounded integration checks; uses no training data or pretrained checkpoints."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["gpu", "wait", "fail", "missing", "timeout"], default="gpu")
args = parser.parse_args()
output = Path("/results")

if args.mode == "gpu":
    import torch

    assert torch.cuda.is_available(), "CUDA must be available inside Docker"
    value = torch.linspace(-1, 1, 1024, device="cuda", requires_grad=True)
    loss = (value * value).mean()
    loss.backward()
    torch.cuda.synchronize()
    result = {"cuda": True, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "loss": loss.item(), "finite_gradients": bool(torch.isfinite(value.grad).all()),
              "literal_environment": os.environ.get("EDGE_LITERAL"), "checkpoint_mount_readable": Path("/checkpoints").is_dir()}
    assert result["finite_gradients"] and result["checkpoint_mount_readable"]
    (output / "gpu.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
elif args.mode == "wait":
    # A descendant ignores TERM, exercising supervisor group kill + Docker exit.
    child = subprocess.Popen([sys.executable, "-u", "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('DESCENDANT-READY',flush=True); time.sleep(110)"])
    (output / "ready.json").write_text(json.dumps({"child_pid": child.pid}))
    print("WAIT-READY", flush=True)
    time.sleep(110)
elif args.mode == "fail":
    print("x" * 70000 + "FAILURE-END", flush=True)
    sys.exit(7)
elif args.mode == "timeout":
    print("TIMEOUT-READY", flush=True)
    time.sleep(30)
else:
    print("Exited successfully without the declared artifact", flush=True)
