"""Render the standard 500k-track evaluation for a completed long-run arm."""
import json
import sys
from pathlib import Path
from ruamel.yaml import YAML
root, arm, backbone = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
yaml = YAML()
repo = Path(__file__).resolve().parents[4]
config = yaml.load((repo / "train/downstream/eval/track_regression_analysis_pretrained.yaml").read_text())
a = config["analysis"]
record = json.loads((root / arm / "artifacts.json").read_text())
a.update(model_yaml=str(root / "experiment.yaml"), model_config=f"{arm}_context",
         checkpoint=record["checkpoint"], output_dir=str(root / arm / "evaluation_500k"),
         training_log=record.get("log_file"), run_name=f"m6_long_{arm}",
         analysis_tag=f"m6_long_{arm}_500k", run_num="42", batch_size=32,
         num_workers=4, max_samples=500000, pretrained_checkpoint=backbone,
         include_sample_metadata=False)
with (root / arm / "analysis.yaml").open("w") as stream:
    yaml.dump(config, stream)
