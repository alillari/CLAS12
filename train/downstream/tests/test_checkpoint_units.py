"""Checkpoint unit declarations, without model or GPU dependencies."""
import unittest

from train.downstream.checkpoint_units import validate_hdf5_momentum_checkpoint_units as validate


class CheckpointUnitsTest(unittest.TestCase):
    def test_saved_gev_contracts_and_historical_physics_settings_are_accepted(self):
        for checkpoint in (
            {"data_contract": {"momentum_unit": "GeV/c"}},
            {"params": {"data_contract": {"momentum_unit": "GeV/c"}}},
            {"params": {"params": {"data_contract": {"momentum_unit": "GeV/c"}}}},
            {"physics_checkpoint_config": {"momentum_scale_to_gev": 1.0}},
            {"params": {"physics_checkpoint": {"momentum_scale_to_gev": 1.0}}},
            {"params": {"params": {"physics_checkpoint": {"momentum_scale_to_gev": 1.0}}}},
        ):
            with self.subTest(checkpoint=checkpoint):
                validate(checkpoint)

    def test_unknown_units_are_not_inferred_from_the_reader_or_stats_path(self):
        for checkpoint in ({}, {"params": {"reader_type": "hdf5", "regression_target_stats": "new_gev.json"}},
                           {"data_contract": None, "physics_checkpoint_config": None}):
            with self.subTest(checkpoint=checkpoint), self.assertRaisesRegex(ValueError, "units are unknown"):
                validate(checkpoint)

    def test_all_explicit_declarations_must_agree_on_gev(self):
        for value in (.001, 1000, 0, None, True, float("nan"), float("inf"), "bad"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "unit mismatch"):
                validate({"data_contract": {"momentum_unit": "GeV/c"},
                          "physics_checkpoint_config": {"momentum_scale_to_gev": value}})
        for checkpoint in (
            {"data_contract": {"momentum_unit": "MeV/c"}},
            {"physics_checkpoint_config": {"momentum_scale_to_gev": 1},
             "params": {"physics_checkpoint": {"momentum_scale_to_gev": .001}}},
            {"params": {"params": {"physics_checkpoint": {"momentum_scale_to_gev": .001}}}},
        ):
            with self.subTest(checkpoint=checkpoint), self.assertRaisesRegex(ValueError, "unit mismatch"):
                validate(checkpoint)

    def test_resolved_saved_params_override_the_original_yaml_copy(self):
        validate({"params": {"physics_checkpoint": {"momentum_scale_to_gev": 1},
                             "params": {"physics_checkpoint": {"momentum_scale_to_gev": .001}}}})


if __name__ == "__main__":
    unittest.main()
