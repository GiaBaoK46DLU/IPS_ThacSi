"""Only shared code stays at the IPS root; KAN owns its implementation."""
import importlib
from pathlib import Path
import unittest

from ips.algorithms.base import Algorithm
from ips.algorithms.kan import KANMixerAlgorithm
from ips.algorithms.kan.paths import EXPERIMENT_CONFIG, MODEL_CONFIG, PACKAGE_DIR, resolve_config_path
from ips.data import ROOT
from ips.experiments.contracts import ExperimentConfig
from ips.experiments.factory import default_factory


class KANLayoutTests(unittest.TestCase):
    def test_no_legacy_aliases_or_personal_code_at_the_shared_root(self):
        root = ROOT/"src/ips"
        self.assertEqual({path.name for path in root.glob("*.py")},
                         {"__init__.py", "__main__.py", "cli.py", "data.py", "metrics.py"})
        self.assertFalse((root/"models").exists())
        self.assertFalse((root/"algorithms/kan_mixer.py").exists())

    def test_configuration_and_factory_point_to_the_kan_folder(self):
        self.assertEqual(KANMixerAlgorithm.__bases__, (Algorithm,))
        implementation = importlib.import_module(KANMixerAlgorithm.__module__)
        self.assertTrue(Path(implementation.__file__).resolve().is_relative_to(PACKAGE_DIR))
        config = ExperimentConfig.from_json(EXPERIMENT_CONFIG)
        self.assertIsInstance(default_factory().create(config.algorithm_id, config), KANMixerAlgorithm)
        self.assertTrue(MODEL_CONFIG.is_file())
        self.assertEqual(resolve_config_path(ROOT/"configs/kan_mixer.json"), MODEL_CONFIG)
        self.assertEqual(resolve_config_path(ROOT/"configs/experiments/kan_mixer_smoke.json"), EXPERIMENT_CONFIG)
        self.assertEqual(resolve_config_path(ROOT/"configs/unknown.json"), ROOT/"configs/unknown.json")

    def test_kan_tests_stay_under_the_implementation_folder(self):
        module = importlib.import_module("ips.algorithms.kan.tests.test_pipeline")
        self.assertTrue(Path(module.__file__).resolve().is_relative_to(PACKAGE_DIR))
