"""Read-only runtime, cluster-data and historical-recipe inspection."""

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch
from mmap_ninja import RaggedMmap

ROOT = Path("/home/alessio/ML-work")
OUT = Path("/results")
report = {"hostname": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda,
          "gpu": torch.cuda.get_device_name(0), "packages": {}, "datasets": {}, "recipes": {}}
for name in ["mamba-ssm", "triton", "causal-conv1d", "mmap-ninja", "cosine-annealing-warmup"]:
    try:
        distribution = importlib.metadata.distribution(name)
        report["packages"][name] = {"version": distribution.version, "source": distribution.read_text("direct_url.json")}
    except importlib.metadata.PackageNotFoundError:
        report["packages"][name] = None
module = importlib.import_module("mamba_ssm.modules.mamba3")
report["mamba3_module"] = {"path": module.__file__, "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
data_root = ROOT / "HIPO_processing/multi_particle_storage/big-data"
for candidate in sorted(data_root.iterdir()):
    if not candidate.is_dir() or "cross" in candidate.name or "mmap" not in candidate.name:
        continue
    entry = {"entries": sorted(p.name for p in candidate.iterdir()), "metadata": {}, "splits": {}}
    for p in sorted(candidate.glob("*.json"))[:5]:
        if p.stat().st_size <= 32000:
            entry["metadata"][p.name] = json.loads(p.read_text())
    for split in ["pretrain", "train", "val", "valid", "test"]:
        path = candidate / ("features_" + split)
        if not path.exists():
            continue
        features = RaggedMmap(str(path))
        samples = []
        for index in sorted({0, len(features) // 2, len(features) - 1}):
            if index >= 0:
                value = np.asarray(features[index])
                samples.append({"index": index, "shape": list(value.shape), "first_rows": value[:2].tolist(), "finite": bool(np.isfinite(value).all())})
        entry["splits"][split] = {"events": len(features), "samples": samples, "mmap_attributes": list(vars(features))}
    report["datasets"][candidate.name] = entry
stats = data_root / "stats"
report["stats_files"] = sorted(p.name for p in stats.iterdir()) if stats.exists() else []
pretrain = ROOT / "result_deep_storage/pretrain"
report["pretrain_studies"] = sorted(p.name for p in pretrain.iterdir()) if pretrain.exists() else []
study = pretrain / "m4_k5_v602_20261001"
if study.exists():
    report["historical_study_files"] = sorted(p.name for p in study.iterdir())
    for p in sorted(study.glob("*.yaml")):
        report["recipes"][str(p.relative_to(ROOT))] = p.read_text()[:10000]
    for p in sorted(study.glob("*.json")):
        if p.stat().st_size < 15000:
            report["recipes"][str(p.relative_to(ROOT))] = json.loads(p.read_text())
    for p in sorted(study.glob("**/model.yaml"))[:2]:
        report["recipes"][str(p.relative_to(ROOT))] = p.read_text()[:10000]
(OUT / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"datasets": {k: {s: d["events"] for s, d in v["splits"].items()} for k, v in report["datasets"].items()}, "runtime": report["packages"]}), flush=True)
