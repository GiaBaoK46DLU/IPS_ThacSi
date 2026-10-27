"""Guards for the frozen V2 replication protocol."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("v2_refit_experiment", Path(__file__).resolve().parents[1] / "scripts/run_kan_v2_refit.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class V2ReplicationTests(unittest.TestCase):
    def test_epoch_rule_requires_every_declared_seed_and_valid_epochs(self):
        rows = [{"seed": s, "best_epoch": e} for s, e in zip(experiment.SEEDS, [21, 27, 37, 30, 28, 40, 20, 25, 35, 26])]
        self.assertEqual(experiment.refit_epochs(rows), 28)
        actual_epochs = [21, 27, 37, 40, 35, 28, 25, 29, 29, 17]
        self.assertEqual(experiment.refit_epochs([
            {"seed": s, "best_epoch": e} for s, e in zip(experiment.SEEDS, actual_epochs)]), 28)
        with self.assertRaises(ValueError):
            experiment.refit_epochs(rows[:-1])
        with self.assertRaises(ValueError):
            experiment.refit_epochs(list(reversed(rows)))
        rows[0]["best_epoch"] = 0
        with self.assertRaises(ValueError):
            experiment.refit_epochs(rows)

    def test_training_design_keeps_algorithmic_changes_visible(self):
        config = experiment.read(experiment.CONFIG)
        fitted = copy.deepcopy(config)
        fitted["positioning"].update(references=[{"class_id": 0, "xy": [0, 0]}], fit_scope="internal_train_only")
        self.assertEqual(experiment.training_design(fitted), config)
        fitted["training"]["sampling"] = "reference_points"
        self.assertNotEqual(experiment.training_design(fitted), config)
        self.assertEqual(config["training"]["patience"], 20)
        self.assertEqual(config["positioning"]["mask_scale"], 0)
        self.assertEqual(config["positioning"]["rssi_noise_db"], 2)

    def test_frozen_artifact_rejects_reselection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            experiment.immutable(path, {"epoch": 27})
            before = path.read_bytes()
            experiment.immutable(path, {"epoch": 27})
            with self.assertRaises(ValueError):
                experiment.immutable(path, {"epoch": 31})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
