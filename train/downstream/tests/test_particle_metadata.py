"""CPU-only PDG and HDF5 track metadata tests, also run in the CI matrix."""
import unittest
from unittest.mock import Mock

import numpy as np

from fm4npp.particle_metadata import metadata_from_pdg
from fm4npp.datasets.cvt_downstream import CVTDownstreamSource


class ParticleMetadataTest(unittest.TestCase):
    def test_supported_species_and_antiparticles(self):
        for pdg, category, label, charge in ((211, 0, 'pion', 1), (321, 1, 'kaon', 1),
                (2212, 2, 'proton', 1), (11, 3, 'electron', -1), (13, 4, 'other', -1),
                (2112, 4, 'other', 0), (22, 4, 'other', 0), (3112, 4, 'other', -1)):
            for sign in (1, -1):
                with self.subTest(pdg=sign*pdg):
                    result = metadata_from_pdg(sign*pdg)
                    self.assertEqual(result['charge'], sign*charge)
                    self.assertEqual((result['pid_class'], result['pid_label']), (category, label))
                    self.assertEqual(result['pid_metadata_status'], 'known')

    def test_ions_use_atomic_number_not_the_sign_alone(self):
        for code, charge in ((1000010020, 1), (1000020040, 2), (-1000020040, -2)):
            self.assertEqual(metadata_from_pdg(code)['charge'], charge)
        self.assertIsNone(metadata_from_pdg(1000030020)['charge'])

    def test_missing_unknown_and_malformed_ids_do_not_invent_a_charge(self):
        for value in (None, 0, -100, np.nan, np.inf, 211.5, True, 'n/a'):
            result = metadata_from_pdg(value)
            self.assertEqual(result['pid_metadata_status'], 'missing')
            self.assertEqual(result['pid_class'], -1)
            self.assertIsNone(result['charge'])
        result = metadata_from_pdg(999999)
        self.assertEqual(result['pdg'], 999999)
        self.assertEqual(result['pid_metadata_status'], 'unknown')
        self.assertIsNone(result['charge'])

    def test_selected_truth_track_excludes_contaminating_candidates(self):
        source = CVTDownstreamSource.__new__(CVTDownstreamSource)
        source.read = Mock(return_value=[np.array([8, 3, 3, 3, -1]),
            np.array([2212, -211, -211, 321, 11]), np.array([1, 1, 1, 0, 1], dtype=bool)])
        result = source.particle_metadata(5, 3)
        self.assertEqual((result['pdg'], result['charge']), (-211, -1))
        self.assertEqual(source.particle_metadata(5, 99)['pid_metadata_status'], 'missing')
        source.read.return_value[2][3] = True
        with self.assertRaisesRegex(ValueError, 'Conflicting PDG'):
            source.particle_metadata(5, 3)


if __name__ == '__main__':
    unittest.main()
