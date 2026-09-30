"""Render a 500k-track physics evaluation for one completed arm."""
import json
import sys
from pathlib import Path
from ruamel.yaml import YAML

root, arm = Path(sys.argv[1]), sys.argv[2]
repo = Path(__file__).resolve().parents[4]
yaml = YAML()
config = yaml.load((repo / "train/downstream/eval/track_regression_analysis_pretrained.yaml").read_text())
analysis = config["analysis"]
record = json.loads((root / arm / "artifacts.json").read_text())
pretrained = arm.startswith("pretrained_")
analysis.update(
    model_yaml=str(root / "experiment.yaml"), model_config=arm,
    checkpoint=record["checkpoint"], output_dir=str(root / arm / "evaluation_500k"),
    training_log=record.get("log_file"), run_name=arm,
    analysis_tag=f"cartesian_factorial_{arm}_500k", run_num="11",
    batch_size=128, num_workers=4, max_samples=500000,
    use_pretrained_backbone=pretrained,
    pretrained_checkpoint=(str(Path("/home/alessio/ML-work/pretrained-FMs/campaign_4/scale_w1536_d12_n39553933/ckpt_best.tar")) if pretrained else None),
    include_sample_metadata=False,
)
with (root / arm / "analysis.yaml").open("w") as stream:
    yaml.dump(config, stream)
