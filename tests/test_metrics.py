"""Shared IPS metrics remain independent of private comparison/report code."""
import unittest
import numpy as np
from ips.metrics import evaluate_predictions


class MetricsTests(unittest.TestCase):
    classes = [{"class_id": 0, "building": 0, "floor": 0},
               {"class_id": 1, "building": 1, "floor": 1}]

    def test_coordinate_mae_and_2d_error_are_distinct_with_inclusive_threshold(self):
        result = evaluate_predictions([0, 1], [0, 0], [[0, 0], [0, 0]], [[3, 4], [0, 8]], self.classes)
        self.assertEqual(result["mean_2d"], 6.5)
        self.assertEqual(result["mae_coordinates"], 3.75)
        self.assertAlmostEqual(result["rmse_2d"], np.sqrt(44.5))
        self.assertEqual(result["thresholds"]["8"]["fraction_2d_within"], 1.)
        self.assertEqual(result["thresholds"]["8"]["fraction_joint_correct_and_2d_within"], .5)
        self.assertEqual(result["conditional_mean_2d_joint_correct"], 5.)

    def test_no_correct_joint_prediction_has_no_conditional_error(self):
        result = evaluate_predictions([0], [1], [[0, 0]], [[3, 4]], self.classes)
        self.assertIsNone(result["conditional_mean_2d_joint_correct"])
        self.assertEqual(result["conditional_count_joint_correct"], 0)

    def test_empty_or_nonfinite_predictions_are_rejected(self):
        for values in (([], [], [], []), ([0], [0], [[0, 0]], [[float("nan"), 0]])):
            with self.assertRaises(ValueError):
                evaluate_predictions(*values, self.classes)
