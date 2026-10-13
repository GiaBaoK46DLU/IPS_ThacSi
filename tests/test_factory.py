"""Cross-backend contracts, split integrity and checkpoint/CLI compatibility."""
import copy
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ips.algorithms.base import Algorithm
from ips.algorithms.huim import HUIMKNNAlgorithm
from ips.algorithms.huim.adapter import HUIMAlgorithm
from ips.algorithms.llm.adapter import LLMAlgorithm
from ips.algorithms.kan import KANMixerAlgorithm
from ips.cli import main
from ips.data import AP_COLUMNS, OUTPUT, ROOT, feature_transform, json_bytes, sha256
from ips.data_access import FingerprintInputs, FingerprintSplit, load_bundle
from ips.algorithms.kan.dataset import FingerprintDataset
from ips.experiments.contracts import AlgorithmExecution, ExperimentConfig, IPSPredictions
from ips.experiments.evaluator import IPSEvaluator
from ips.experiments.factory import AlgorithmFactory, AlgorithmNotImplementedError, UnknownAlgorithmError, default_factory
from ips.experiments.runner import ExperimentRunner
from ips.metrics import evaluate_predictions, inverse_xy
from ips.algorithms.kan.models import build_model
from ips.algorithms.kan.runtime import read_config


class TrainCentroidFixture(Algorithm):
    """Test-only non-neural executor; no production registration."""
    events = []
    finalized = True

    def prepare(self, train):
        self.events.append(("prepare", train.name))
        self.center = train.xy.mean(axis=0)
        self.label = int(train.class_ids[0])

    def execute(self, validation):
        self.events.append(("execute", validation.name))
        return AlgorithmExecution(finalized=self.finalized, run_type="test_fixture")

    def predict(self, inputs):
        self.events.append(("predict", inputs.source_id))
        assert not hasattr(inputs, "xy") and not hasattr(inputs, "class_ids")
        return IPSPredictions(inputs.row_indices, inputs.source_id,
                              xy=np.tile(self.center, (len(inputs), 1)),
                              class_ids=np.full(len(inputs), self.label, dtype=np.int64))


class FactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(OUTPUT)
        cls.train = cls.bundle.split("train", 26)
        cls.validation = cls.bundle.split("validation", 26)

    def config(self, name="kan_mixer", **kwargs):
        return ExperimentConfig(name, ROOT/"runs/unit_factory_not_executed", **kwargs)

    def predictions(self, **kwargs):
        split = self.validation
        return IPSPredictions(split.inputs.row_indices, split.inputs.source_id, **kwargs)

    def test_registry_creates_only_implemented_algorithms(self):
        factory = default_factory()
        self.assertEqual(factory.available_ids(), ("huim", "kan_mixer"))
        self.assertEqual(KANMixerAlgorithm.__bases__, (Algorithm,))
        for name, implementation in (("kan_mixer", KANMixerAlgorithm), ("huim", HUIMKNNAlgorithm)):
            algorithm = factory.create(name, self.config(name))
            self.assertIsInstance(algorithm, implementation)
            self.assertIsInstance(algorithm, Algorithm)
            self.assertNotIsInstance(algorithm, torch.nn.Module)
        for name in ("llm", "fastkan", "s_kan"):
            with self.assertRaisesRegex(AlgorithmNotImplementedError, "not implemented|no .*implementation"):
                factory.create(name, self.config(name))
        with self.assertRaisesRegex(UnknownAlgorithmError, "Unregistered.*Available"):
            factory.create("typo", self.config("typo"))
        with self.assertRaisesRegex(ValueError, "differs"):
            factory.create("another_id", self.config())
        for name in ("mlp", "mlp_mixer"):
            with self.assertRaisesRegex(UnknownAlgorithmError, "personal study"):
                factory.create(name, self.config(name))

    def test_registry_rejects_duplicates_modules_and_abstract_adapters(self):
        factory = default_factory()
        with self.assertRaisesRegex(ValueError, "already registered"):
            factory.register("kan_mixer", KANMixerAlgorithm)
        for implementation in (Algorithm, LLMAlgorithm, HUIMAlgorithm, torch.nn.Linear):
            with self.assertRaises(TypeError):
                factory.register("invalid", implementation)

    def test_shared_data_is_torch_independent(self):
        output = subprocess.check_output([sys.executable, "-c",
            "import sys; import ips.data_access; from ips.experiments.runner import ExperimentRunner; "
            "assert 'torch' not in sys.modules; "
            "assert not hasattr(ips.data_access.Bundle, 'dataset'); print('PASS')"], cwd=ROOT, text=True)
        self.assertIn("PASS", output)

    def test_shared_split_preserves_original_indices_labels_aps_and_tensors(self):
        self.assertEqual({s: len(v) for s, v in self.bundle.indices.items()},
                         {"train": 16025, "validation": 3912, "test": 1111})
        self.assertFalse(set(self.bundle.indices["train"]) & set(self.bundle.indices["validation"]))
        for name in ("train", "validation", "test"):
            split = self.bundle.split(name, 26)
            original = self.bundle.development if name != "test" else None
            self.assertEqual(split.inputs.ap_order, tuple(AP_COLUMNS))
            self.assertTrue(np.all(np.diff(split.inputs.row_indices) > 0))
            self.assertFalse(split.inputs.rssi.flags.writeable)
            tensor = FingerprintDataset.from_split(split)
            np.testing.assert_array_equal(tensor.indices, split.inputs.row_indices)
            np.testing.assert_array_equal(tensor.classes, split.class_ids)
            np.testing.assert_array_equal(tensor.xy_original, split.xy)
            np.testing.assert_array_equal(tensor.features, feature_transform(split.inputs.rssi))
            np.testing.assert_allclose(inverse_xy(tensor.xy.numpy(), split.info.preprocessing),
                                       split.xy, atol=1e-5, rtol=0)
            if original is not None:
                reference = FingerprintDataset(original, split.inputs.row_indices.copy(),
                                               split.info.preprocessing, split.info.classes)
                torch.testing.assert_close(tensor.features, reference.features, rtol=0, atol=0)
                torch.testing.assert_close(tensor.xy, reference.xy, rtol=0, atol=0)
                np.testing.assert_array_equal(split.metadata["USERID"],
                    original.column("USERID")[split.inputs.row_indices])
        # Historical per-class quota selection, followed by original CSV row order.
        pairs = self.bundle.development.pairs
        expected = sorted(int(next(i for i in self.bundle.indices["train"]
            if pairs[i] == (c["building"], c["floor"]))) for c in self.bundle.classes)
        np.testing.assert_array_equal(self.bundle.split("train", 13).inputs.row_indices, expected)

    def test_saved_preprocessing_rejects_validation_fitted_center_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"split"
            shutil.copytree(OUTPUT, folder)
            preprocessing = copy.deepcopy(self.bundle.preprocessing)
            preprocessing["coordinates"]["center"] = self.validation.xy.mean(axis=0).tolist()
            (folder/"preprocessing.json").write_bytes(json_bytes(preprocessing))
            manifest = copy.deepcopy(self.bundle.manifest)
            manifest["artifact_sha256"]["preprocessing.json"] = sha256(folder/"preprocessing.json")
            (folder/"manifest.json").write_bytes(json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "train|Train|coordinate|Coordinate"):
                load_bundle(folder)

    def test_evaluator_matches_legacy_and_does_not_invent_missing_outputs(self):
        split = self.validation
        xy = split.xy + np.array([3., 4.])
        expected = evaluate_predictions(split.class_ids, split.class_ids, split.xy, xy, split.info.classes)
        metrics, matrices = IPSEvaluator().evaluate(split, self.predictions(xy=xy, class_ids=split.class_ids), "position")
        self.assertEqual(metrics, expected)
        self.assertEqual(len(matrices), 3)
        self.assertEqual(metrics["mean_2d"], 5.)
        coordinates, no_matrices = IPSEvaluator().evaluate(split, self.predictions(xy=xy), "coordinates")
        self.assertEqual(no_matrices, {})
        self.assertNotIn("joint_accuracy", coordinates)
        for name in ("mae_xy", "rmse_xy", "mae_coordinates", "rmse_coordinates", "mean_2d", "p90_2d"):
            self.assertEqual(coordinates[name], expected[name])
        classified, matrices = IPSEvaluator().evaluate(split, self.predictions(class_ids=split.class_ids), "classification")
        self.assertNotIn("mean_2d", classified)
        self.assertEqual(classified["joint_macro_f1"], expected["joint_macro_f1"])
        self.assertEqual(len(matrices), 3)
        buildings = np.array([c["building"] for c in split.info.classes])[split.class_ids]
        only_building, matrices = IPSEvaluator().evaluate(split, self.predictions(buildings=buildings))
        self.assertEqual(list(matrices), ["building_confusion_matrix"])
        self.assertNotIn("floor_accuracy", only_building)

    def test_prediction_contract_rejects_misalignment_invalid_values_and_wrong_task(self):
        split = self.validation
        invalid = [self.predictions(), self.predictions(xy=np.full_like(split.xy, np.nan)),
                   self.predictions(class_ids=np.full(len(split), 13)),
                   self.predictions(class_ids=split.class_ids.astype(float)),
                   self.predictions(class_ids=split.class_ids, confidence=np.full(len(split), 2.)),
                   IPSPredictions(split.inputs.row_indices[::-1], split.inputs.source_id, xy=split.xy),
                   IPSPredictions(split.inputs.row_indices, "another.csv", xy=split.xy)]
        for prediction in invalid:
            with self.assertRaises(ValueError):
                IPSEvaluator().evaluate(split, prediction)
        with self.assertRaisesRegex(ValueError, "position requires"):
            IPSEvaluator().evaluate(split, self.predictions(xy=split.xy), "position")

    def test_pair_predictions_convert_to_shared_class_ids(self):
        pairs = np.asarray([(c["building"], c["floor"]) for c in self.bundle.classes])[self.validation.class_ids]
        prediction = self.predictions(buildings=pairs[:, 0], floors=pairs[:, 1])
        prediction.validate(self.validation.inputs, self.bundle.classes)
        np.testing.assert_array_equal(prediction.class_ids, self.validation.class_ids)

    def test_runner_non_neural_execution_exports_and_delays_test_until_finalization(self):
        for finalized in (True, False):
            with self.subTest(finalized=finalized), tempfile.TemporaryDirectory() as tmp:
                events = []
                TrainCentroidFixture.events = events
                TrainCentroidFixture.finalized = finalized
                real_bundle = self.bundle
                class SmallBundle:
                    manifest, preprocessing, classes, ap_order = (real_bundle.manifest, real_bundle.preprocessing,
                                                                 real_bundle.classes, real_bundle.ap_order)
                    def split(inner, name, limit=None):
                        events.append(("split", name))
                        return real_bundle.split(name, 13)
                class Repository:
                    def load(inner, directory):
                        return SmallBundle()
                factory = AlgorithmFactory()
                factory.register("unit_centroid", TrainCentroidFixture)
                config = ExperimentConfig("unit_centroid", Path(tmp)/"run", evaluation_splits=("validation", "test"),
                                          export_plots=False)
                runner = ExperimentRunner(factory=factory, repository=Repository())
                with patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                    if not finalized:
                        with self.assertRaisesRegex(ValueError, "finalized"):
                            runner.run(config)
                        self.assertNotIn(("split", "test"), events)
                        self.assertTrue((config.output_dir/"failure.json").exists())
                        continue
                    result = runner.run(config)
                self.assertLess(events.index(("execute", "validation")), events.index(("split", "test")))
                self.assertEqual(events[2], ("prepare", "train"))
                self.assertIsNone(result.history)
                self.assertIsNone(result.resources)
                self.assertIsNone(result.diagram)
                self.assertGreater(result.metrics["test"]["mean_2d"], 0)
                with result.artifacts["test_predictions"].open(encoding="utf-8", newline="") as stream:
                    records = list(csv.DictReader(stream))
                self.assertEqual([int(r["row_index"]) for r in records],
                                 result.predictions["test"].row_indices.tolist())
                self.assertTrue(result.artifacts["run_result"].exists())
                self.assertTrue(result.log_path.exists())
                with self.assertRaises(FileExistsError):
                    runner.run(config)

    def test_smoke_test_and_unknown_config_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Smoke"):
            self.config(smoke=True, evaluation_splits=("test",))
        for params in ({"threshold": float("nan")}, {"model": object()}):
            with self.assertRaisesRegex(ValueError, "JSON-serializable"):
                self.config(params=params)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"config.json"
            path.write_text('{"algorithm_id":"mlp", "output_dir":"runs/x", "surprise":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown experiment fields"):
                ExperimentConfig.from_json(path)
            config = ExperimentConfig("not_registered", Path(tmp)/"not_created")
            with self.assertRaises(UnknownAlgorithmError):
                ExperimentRunner().run(config)
            self.assertFalse(config.output_dir.exists())

    def test_checkpoint_v1_fixture_is_loaded_by_algorithm_with_identical_predictions(self):
        settings = read_config(ROOT/"src/ips/algorithms/kan/configs/model.json")
        model = build_model(settings, len(self.bundle.classes)).eval()
        info = self.train.info
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"v1.pt"
            torch.save({"format_version": 1, "config": settings, "model_state_dict": model.state_dict(),
                        "run_type": "training", "preprocessing": info.preprocessing, "classes": info.classes,
                        "ap_order": info.ap_order, "split_manifest_sha256": sha256(OUTPUT/"manifest.json")}, path)
            algorithm = default_factory().create("kan_mixer", self.config(params={"checkpoint": str(path)}))
            algorithm.prepare(self.train)
            self.assertTrue(algorithm.execute(self.validation).finalized)
            prediction = algorithm.predict(self.validation.inputs).validate(self.validation.inputs, info.classes)
            with torch.inference_mode():
                output = model(FingerprintDataset.from_split(self.validation).features)
            np.testing.assert_array_equal(prediction.class_ids, output["logits"].argmax(dim=1).numpy())
            np.testing.assert_allclose(prediction.xy, inverse_xy(output["xy"].numpy(), info.preprocessing), atol=1e-6)
            payload = torch.load(path, weights_only=True)
            payload["ap_order"] = list(reversed(info.ap_order))
            torch.save(payload, path)
            wrong = default_factory().create("kan_mixer", self.config(params={"checkpoint": str(path)}))
            wrong.prepare(self.train)
            with self.assertRaisesRegex(ValueError, "Checkpoint algorithm"):
                wrong.execute(self.validation)

    def test_legacy_cli_preserves_train_predict_evaluate_and_check_options(self):
        with patch("ips.algorithms.kan.train.train") as training:
            main(["train", "--device", "cpu", "--epochs", "1", "--smoke", "--run-dir", "runs/unit"])
            self.assertEqual(training.call_args.args[0]["model"], "kan_mixer")
            self.assertEqual(training.call_args.args[-2:], (1, True))
        with patch("ips.algorithms.kan.predict.predict") as prediction:
            main(["predict", "--checkpoint", "v1.pt", "--csv", "data.csv", "--output", "pred.csv", "--device", "cpu"])
            self.assertEqual(prediction.call_args.args, (Path("v1.pt"), Path("data.csv"), Path("pred.csv"), "cpu"))
        with patch("ips.algorithms.kan.predict.evaluate") as evaluation:
            main(["evaluate", "--checkpoint", "v1.pt", "--output-dir", "runs/unit", "--device", "cpu"])
            self.assertEqual(evaluation.call_args.args[2], "validation")
        with patch("sys.stdout", new=io.StringIO()) as output:
            main(["check", "--device", "cpu"])
            self.assertIn("PASS: data integrity and forward shapes", output.getvalue())
            self.assertNotIn("mlp", output.getvalue())
        with patch("sys.stdout", new=io.StringIO()) as output:
            main(["algorithms"])
            self.assertIn("Implemented: huim, kan_mixer", output.getvalue())
            self.assertNotIn("mlp", output.getvalue())
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            main(["train", "--model", "mlp_mixer", "--run-dir", "runs/unit"])


if __name__ == "__main__":
    unittest.main()
