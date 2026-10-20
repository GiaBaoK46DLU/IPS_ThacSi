"""Known-building experts must retain frozen source rows and fit train only."""
import unittest
from pathlib import Path

import numpy as np

from ips.data import AP_COLUMNS
from ips.data_access.contracts import DatasetInfo, FingerprintInputs, FingerprintSplit
from ips.algorithms.kan.uji_buildings import fit_coordinates, partition


class BuildingPartitionTests(unittest.TestCase):
    def split(self):
        classes = [{"class_id": i, "building": b, "floor": f}
                   for i, (b, f) in enumerate([(0, 0), (0, 1), (1, 0), (2, 0)])]
        info = DatasetInfo(Path('.'), {}, {}, classes, AP_COLUMNS.copy())
        inputs = FingerprintInputs(np.full((8, 520), -60), np.arange(8)*3,
                                   'original_training.csv')
        return FingerprintSplit('train', inputs, np.tile(np.arange(4), 2),
            np.column_stack([np.arange(8), np.arange(8)+10]),
            {'USERID': np.arange(8), 'PHONEID': np.arange(8)+1}, info)

    def test_buildings_are_disjoint_cover_original_and_keep_row_order(self):
        original = self.split()
        all_indices = []
        for building in (0, 1, 2):
            classes = [{**c, 'class_id': i} for i, c in enumerate(
                c for c in original.info.classes if c['building'] == building)]
            info = DatasetInfo(Path('.'), {}, {}, classes, AP_COLUMNS.copy())
            subset = partition(original, building, info)
            selected = np.asarray([original.info.classes[c]['building'] == building
                                   for c in original.class_ids])
            np.testing.assert_array_equal(subset.inputs.row_indices, original.inputs.row_indices[selected])
            np.testing.assert_array_equal(subset.xy, original.xy[selected])
            np.testing.assert_array_equal(subset.metadata['USERID'], original.metadata['USERID'][selected])
            self.assertEqual(subset.inputs.source_id, original.inputs.source_id)
            self.assertEqual(subset.inputs.ap_order, original.inputs.ap_order)
            self.assertTrue((subset.class_ids < len(classes)).all())
            all_indices.extend(subset.inputs.row_indices)
        np.testing.assert_array_equal(np.sort(all_indices), original.inputs.row_indices)
        self.assertEqual(len(set(all_indices)), len(original))

    def test_coordinate_fit_uses_training_points_and_shared_axis_scale(self):
        train = np.asarray([[0., 10.], [2., 14.]])
        fitted = fit_coordinates(train)
        np.testing.assert_allclose(fitted['center'], [1., 12.])
        self.assertAlmostEqual(fitted['shared_scale'], np.sqrt(2.5))
        heldout = np.asarray([[1000., 2000.]])
        normalized = (heldout - fitted['center'])/fitted['shared_scale']
        np.testing.assert_allclose(normalized*fitted['shared_scale']+fitted['center'], heldout)
        np.testing.assert_array_equal(fit_coordinates(train)['center'], fitted['center'])


if __name__ == '__main__':
    unittest.main()
