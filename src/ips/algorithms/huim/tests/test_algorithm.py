"""HUIM lifecycle on small deterministic subsets of the shared split: train-only fitting,
validation-only selection, prediction contract, unmatched queries and Factory/Runner use."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ips.algorithms.huim import HUIMAlgorithm, HUIMKNNAlgorithm
from ips.algorithms.huim.positioning import Positioner, KNNConfig, representation
from ips.algorithms.huim.utility import TransactionEncoder, external_utility, ranking, states
from ips.data import OUTPUT, ROOT
from ips.data_access import FingerprintInputs, FingerprintSplit, load_bundle
from ips.experiments.contracts import ExperimentConfig
from ips.experiments.evaluator import IPSEvaluator
from ips.experiments.factory import default_factory
from ips.experiments.runner import ExperimentRunner

# The smoke subsets repeat scans of the same points, so HUIs explode just below 10% of the
# total utility: the 5% run stops at max_huis and its (truncated) union is the AP candidate.
FAST = {"min_util_pct": [20.0, 10.0, 5.0], "max_bisections": 0, "time_limit_s": 120.0,
        "max_huis": 5000, "topn_budgets": [5, 10]}


class HUIMAlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(OUTPUT)
        cls.train = cls.bundle.split("train", 260)
        cls.validation = cls.bundle.split("validation", 130)

    def create(self, **params):
        config = ExperimentConfig("huim", ROOT/"runs/unit_huim_not_executed", params={**FAST, **params})
        return default_factory().create("huim", config)

    def fitted(self, **params):
        algorithm = self.create(**params)
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stdout", new=io.StringIO()):
            object.__setattr__(algorithm.config, "output_dir", Path(tmp))
            algorithm.prepare(self.train)
            execution = algorithm.execute(self.validation)
        return algorithm, execution

    def test_encoder_maps_missing_to_absent_items_and_rescales_on_train_range(self):
        encoder = TransactionEncoder.fit(np.asarray([[-100, 100, -40]], dtype=np.int16))
        self.assertEqual((encoder.rssi_min, encoder.rssi_max), (-100.0, -40.0))
        q = encoder.quantities(np.asarray([[-100, 100, -40, -70, -120]], dtype=np.int16))
        # -70: 1 + (30/60)*99 = 50.5 -> 50 (round half to even); below the train range clips to 1
        np.testing.assert_array_equal(q, [[1, 0, 100, 50, 1]])
        self.assertEqual(encoder.transactions(np.asarray([[100, -40, 100]]))[0], {1: 100})
        np.testing.assert_array_equal(states(np.asarray([[100, -104, -85, -60, 0]])), [[0, 1, 2, 4, 5]])

    def test_external_utilities_are_positive_and_fitted_on_labels(self):
        rssi, labels = self.train.inputs.rssi, self.train.class_ids
        a, _ = external_utility("A", rssi, labels)
        np.testing.assert_array_equal(a, np.ones(520))
        for model in ("B", "C"):
            p, info = external_utility(model, rssi, labels)
            self.assertTrue((p > 0).all() and p.max() == 1.0, model)
            self.assertEqual(info["model"], model)
        with self.assertRaises(ValueError):
            external_utility("D", rssi, labels)
        self.assertEqual(ranking([(("x", "y"), 10.0, 1), (("y",), 1.0, 1)]), ["y", "x"])

    def test_factory_registers_the_concrete_implementation_only(self):
        self.assertIn("huim", default_factory().available_ids())
        self.assertEqual(HUIMKNNAlgorithm.__bases__, (HUIMAlgorithm,))
        self.assertIsInstance(self.create(), HUIMKNNAlgorithm)
        with self.assertRaisesRegex(ValueError, "Unknown HUIM params"):
            self.create(surprise=1)
        with self.assertRaisesRegex(ValueError, "utility_model"):
            self.create(utility_model="Z")

    def test_lifecycle_freezes_on_validation_and_returns_a_valid_position(self):
        algorithm = self.create()
        with patch("sys.stdout", new=io.StringIO()):
            algorithm.prepare(self.train)
        with self.assertRaisesRegex(RuntimeError, "execute"):
            algorithm.predict(self.validation.inputs)
        algorithm, execution = self.fitted()
        self.assertTrue(execution.finalized)
        self.assertEqual([h["stage"] for h in execution.history][:3], ["S0", "S1", "S2"])
        json.dumps(execution.history, allow_nan=False)
        json.dumps(execution.resources, allow_nan=False)
        prediction = algorithm.predict(self.validation.inputs)
        metrics, _ = IPSEvaluator().evaluate(self.validation, prediction, "position")
        self.assertGreater(metrics["joint_accuracy"], 0.5)
        # selection scored exactly the pipeline predict() runs, fallback included
        self.assertAlmostEqual(execution.history[-1]["validation_mean_error"], metrics["mean_2d"], places=5)
        again, _ = self.fitted()
        repeat = again.predict(self.validation.inputs)
        np.testing.assert_array_equal(repeat.xy, prediction.xy)
        np.testing.assert_array_equal(repeat.class_ids, prediction.class_ids)

    def test_selection_never_sees_test_and_rejects_leaking_splits(self):
        algorithm = self.create()
        with patch("sys.stdout", new=io.StringIO()):
            with self.assertRaisesRegex(ValueError, "train split"):
                algorithm.prepare(self.validation)
            algorithm.prepare(self.train)
        with self.assertRaisesRegex(ValueError, "never test"):
            algorithm.execute(self.bundle.split("test", 13))
        overlapping = FingerprintSplit("validation", self.train.inputs, self.train.class_ids, self.train.xy,
                                       {}, self.train.info)
        with self.assertRaisesRegex(ValueError, "overlap"):
            algorithm.execute(overlapping)

    def test_query_without_any_selected_ap_uses_the_train_fallback(self):
        algorithm, _ = self.fitted()
        rssi = np.array(self.validation.inputs.rssi)
        rssi[0, algorithm._columns] = 100
        inputs = FingerprintInputs(rssi, self.validation.inputs.row_indices, self.validation.inputs.source_id)
        prediction = algorithm.predict(inputs)
        self.assertGreaterEqual(algorithm.last_unmatched, 1)
        self.assertTrue(np.isfinite(prediction.xy).all())
        expected = Positioner(self.train.inputs.rssi, self.train.xy, self.train.class_ids,
                              len(self.train.info.classes), algorithm._detected, algorithm._final).predict(rssi[:1])
        np.testing.assert_allclose(prediction.xy[0], expected["xy"][0])

    def test_fallback_none_keeps_the_pure_huim_vector(self):
        algorithm, execution = self.fitted(fallback="none")
        rssi = np.array(self.validation.inputs.rssi)
        rssi[0, algorithm._columns] = 100
        inputs = FingerprintInputs(rssi, self.validation.inputs.row_indices, self.validation.inputs.source_id)
        prediction = algorithm.predict(inputs)
        self.assertGreaterEqual(algorithm.last_unmatched, 1)
        expected = Positioner(self.train.inputs.rssi, self.train.xy, self.train.class_ids,
                              len(self.train.info.classes), algorithm._columns, algorithm._final).predict(rssi[:1])
        np.testing.assert_allclose(prediction.xy[0], expected["xy"][0])
        self.assertEqual(execution.resources["fallback"], "none")
        with self.assertRaisesRegex(ValueError, "fallback"):
            self.create(fallback="centroid")

    def test_neighbour_order_breaks_exact_distance_ties_by_train_row(self):
        reference = np.asarray([[-50, 100], [-50, 100], [-60, 100]], dtype=np.int16)
        xy = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        positioner = Positioner(reference, xy, np.zeros(3, dtype=np.int64), 1, np.arange(2),
                                KNNConfig("positive", "manhattan", 1, 0))
        np.testing.assert_array_equal(positioner.predict(np.asarray([[-50, 100]]))["xy"], [[0.0, 0.0]])
        self.assertEqual(representation(np.asarray([[100]]), "exponential")[0, 0], 0.0)

    def test_runner_smoke_exports_selection_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig("huim", Path(tmp)/"run", params=FAST, smoke=True, export_plots=False)
            with patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                result = ExperimentRunner().run(config)
            self.assertEqual(result.run_type, "smoke_check")
            for name in ("huim_selection", "huim_selection_grid", "run_result", "validation_predictions"):
                self.assertTrue(Path(result.artifacts[name]).exists(), name)
            selection = json.loads(Path(result.artifacts["huim_selection"]).read_text(encoding="utf-8"))
            self.assertTrue(all(ap.startswith("WAP") for ap in selection["selected_aps"]))
            self.assertIn("mean_2d", result.metrics["validation"])


if __name__ == "__main__":
    unittest.main()
