import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from train.downstream.tuning.analyze_track_regression_optuna import (
    infer_max_step_budget,
    validation_history,
)


class OptunaAnalysisTest(unittest.TestCase):
    def test_multi_seed_curves_keep_common_steps_and_objective(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = []
            for seed, contents in [
                (11, "Step\tVal_Loss\n100\t0.2\n200\t0.1\n300\t0.05\n"),
                (17, "Step\tVal_Loss\n100\t0.6\n200\t0.3\n"),
                (23, "Step\tVal_Loss\n100\t0.4\n200\t0.2\n"),
            ]:
                log = Path(temporary) / f"seed_{seed}.log"
                log.write_text(contents)
                results.append({"seed": seed, "log_file": str(log)})
            trial = SimpleNamespace(number=0, intermediate_values={}, user_attrs={
                "seed_results": results, "seed_objective": "mean",
                "execution_contract": {"max_optimizer_steps": 1000},
            })
            self.assertEqual(set(validation_history(trial, None)), {100, 200})
            self.assertAlmostEqual(validation_history(trial, None)[100], 0.4)
            trial.user_attrs["seed_objective"] = "median"
            self.assertAlmostEqual(validation_history(trial, None)[200], 0.2)
            trial.user_attrs["seed_objective"] = "mean_plus_std"
            self.assertAlmostEqual(validation_history(trial, None)[200],
                                   0.2 + (0.02 / 3) ** 0.5)
            # Budget comes from the protocol, not a seed's observed/best step.
            self.assertEqual(infer_max_step_budget([trial], None), 1000)
            Path(results[-1]["log_file"]).unlink()
            with self.assertLogs("optuna-study-analysis", level="WARNING"):
                self.assertEqual(validation_history(trial, None), {})

    def test_legacy_and_pruned_single_seed_histories_remain_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "legacy.log"
            log.write_text("Step\tVal_Loss\n100\t0.5\n200\t0.25\n")
            trial = SimpleNamespace(number=0, intermediate_values={},
                                    user_attrs={"log_file": str(log)})
            self.assertEqual(validation_history(trial, None), {100: 0.5, 200: 0.25})
            trial.intermediate_values = {100: 0.4, 200: float("nan")}
            self.assertEqual(validation_history(trial, None), {100: 0.4})


if __name__ == "__main__":
    unittest.main()
