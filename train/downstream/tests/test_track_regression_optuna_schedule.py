import sys
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from train.downstream.tuning.run_track_regression_optuna import (
    fixed_overrides,
    parse_trial_seeds,
    scheduler_first_cycle_steps,
    validate_study_contract,
)


def args_for_schedule(max_steps=30000, n_cycles=1):
    return Namespace(
        max_optimizer_steps=max_steps,
        n_cycles=n_cycles,
        val_interval_steps=1000,
        early_stopping_min_steps=3000,
        early_stopping_patience=20,
        max_val_batches=500,
        max_train_batches=None,
        num_data_workers=None,
        data_root=None,
        data_root_test=None,
        stat_dir=None,
        regression_target_stats=None,
        eventnumber=100000,
    )


class TrackRegressionOptunaScheduleTest(unittest.TestCase):
    def test_cycle_lengths_are_exact_partitions(self):
        self.assertEqual(scheduler_first_cycle_steps(30000, 1), 30000)
        self.assertEqual(scheduler_first_cycle_steps(30000, 2), 15000)
        self.assertEqual(scheduler_first_cycle_steps(30000, 3), 10000)

    def test_non_integral_cycle_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            scheduler_first_cycle_steps(30000, 7)

    def test_fixed_overrides_use_requested_cycle_count(self):
        overrides = fixed_overrides(args_for_schedule(n_cycles=3))
        self.assertEqual(overrides["max_optimizer_steps"], 30000)
        self.assertEqual(overrides["scheduler_first_cycle_steps"], 10000)

    def test_trial_seeds_are_nonempty_and_deduplicated(self):
        self.assertEqual(parse_trial_seeds("11,17,11,23"), (11, 17, 23))
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_trial_seeds(" , ")

    def test_study_contract_rejects_incompatible_resume(self):
        class Study:
            def __init__(self):
                self.user_attrs = {}
                self.trials = []

            def set_user_attr(self, key, value):
                self.user_attrs[key] = value

        study = Study()
        contract = {"n_cycles": 1, "trial_seeds": [11, 17, 23]}
        validate_study_contract(study, contract)
        validate_study_contract(study, contract)
        with self.assertRaisesRegex(ValueError, "different execution contract"):
            validate_study_contract(study, {"n_cycles": 3, "trial_seeds": [11, 17, 23]})


if __name__ == "__main__":
    unittest.main()
