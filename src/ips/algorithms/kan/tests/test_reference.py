"""Catalogue fitting, label-free decoding and legacy checkpoint coexistence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ips.data import ROOT, META_COLUMNS, sha256
from ips.algorithms.kan.models import build_model
from ips.algorithms.kan.models.reference import fit_reference_points
from ips.algorithms.kan.models.refinement import RefinedKANMixer, db_features
from ips.algorithms.kan.models.ensemble import EnsembleReferenceKANMixer
from ips.algorithms.kan.runtime import configure_runtime, load_checkpoint, read_config
from ips.algorithms.kan.train import combined_loss


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_runtime(42, "cpu", 2)
        cls.base = read_config(ROOT / "src/ips/algorithms/kan/configs/accuracy_reference.json")

    def fitted_config(self):
        config = copy.deepcopy(self.base)
        config["mixer"].update(channels=8, blocks=1, token_hidden=8, channel_hidden=8, dropout=0)
        config["positioning"].update(head_width=8, rssi_noise_db=0, device_shift_db=0, ap_dropout=0)
        classes = torch.arange(13).repeat_interleave(2)
        original = np.column_stack([np.arange(26), np.arange(26) + 4]).astype(np.float64)
        dataset = SimpleNamespace(split_name="train", classes=classes, xy_original=original,
                                  xy=torch.tensor(original / 100, dtype=torch.float32))
        fit_reference_points(config, dataset)
        return config, dataset

    def test_catalogue_is_fit_only_from_supplied_train_and_preserves_class_xy(self):
        config, train = self.fitted_config()
        entries = config["positioning"]["references"]
        self.assertEqual(config["positioning"]["fit_scope"], "internal_train_only")
        self.assertEqual(len(entries), 26)
        np.testing.assert_allclose([p["xy"] for p in entries], train.xy.numpy(), rtol=0, atol=0)
        self.assertEqual([p["class_id"] for p in entries], train.classes.tolist())
        # A second dataset is not reachable from the fitting API.
        self.assertFalse(any(999 in p["xy"] for p in entries))
        heldout = copy.deepcopy(train)
        heldout.split_name = "validation"
        with self.assertRaisesRegex(ValueError, "identified train split"):
            fit_reference_points(config, heldout)
        unfitted = copy.deepcopy(self.base)
        with self.assertRaisesRegex(ValueError, "fitted on internal train"):
            build_model(unfitted)

    def test_decoder_returns_finite_ips_outputs_inside_predicted_floor_hull(self):
        config, _ = self.fitted_config()
        model = build_model(config).eval()
        x = torch.rand(4, 520, 2)
        x[..., 1] = (x[..., 1] > .5).float()
        with torch.inference_mode():
            output = model(x)
        self.assertEqual(output["logits"].shape, (4, 13))
        self.assertEqual(output["xy"].shape, (4, 2))
        self.assertTrue(torch.isfinite(output["xy"]).all())
        probabilities = output["reference_logits"].softmax(1)
        for c in range(13):
            expected = probabilities[:, model.reference_classes == c].sum(1)
            torch.testing.assert_close(output["logits"].softmax(1)[:, c], expected)
        for row, cls in zip(output["xy"], output["logits"].argmax(1)):
            points = model.reference_xy[model.reference_classes == cls]
            self.assertTrue((row >= points.min(0).values - 1e-6).all())
            self.assertTrue((row <= points.max(0).values + 1e-6).all())
        # forward has no building, floor, coordinate or device-ID arguments.
        with self.assertRaises(TypeError):
            model(x, torch.arange(4))

    def test_reference_loss_has_finite_backbone_and_head_gradients(self):
        config, data = self.fitted_config()
        for sigma in (0., .05):
            config["loss"]["reference_sigma_normalized"] = sigma
            model = build_model(config)
            x = torch.rand(4, 520, 2)
            output = model(x)
            loss, _, _ = combined_loss(output, data.classes[:4], data.xy[:4], config["loss"])
            loss.backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_fitted_reference_checkpoint_roundtrip_requires_no_training_csv(self):
        config, _ = self.fitted_config()
        original = build_model(config).eval()
        x = torch.rand(2, 520, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.pt"
            torch.save({"format_version": 1, "config": config, "classes": [{}] * 13,
                        "model_state_dict": original.state_dict()}, path)
            restored, _ = load_checkpoint(path, torch.device("cpu"))
            with torch.inference_mode():
                first, second = original(x), restored(x)
            torch.testing.assert_close(first["xy"], second["xy"], rtol=0, atol=0)
            torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=0)

    def test_db_transform_keeps_missing_and_weak_signal_distinct(self):
        from ips.data import feature_transform
        features = torch.from_numpy(feature_transform(np.array([[-52, -104, 100, 0]], dtype=np.int16)))
        torch.testing.assert_close(db_features(features), torch.tensor([[53/105, 1/105, 0., 1.]]))

    def test_hybrid_uses_predicted_floor_and_preserves_neural_contribution(self):
        class Backbone(torch.nn.Module):
            classes = 2

            def forward(self, features):
                return {"logits": torch.tensor([[-10., 10.]]).expand(len(features), -1),
                        "conditional_xy": torch.tensor([[[4., 4.], [100., 100.]]]).expand(len(features), -1, -1)}

        features = torch.zeros(1, 520, 2)
        features[0, 0] = torch.tensor([.5, 1.])
        signals = db_features(features).repeat(2, 1)
        signals[1] += 1
        payload = {"fit_scope": "internal_train_only", "signal": signals,
                   "xy": torch.tensor([[0., 0.], [200., 200.]]), "class_ids": torch.tensor([0, 1]),
                   "settings": {"neighbors": 1, "distance_power": 2.,
                                "neighbor_class_weight": 1., "neural_coordinate_weight": .25}}
        model = RefinedKANMixer(Backbone(), payload)
        result = model(features)
        self.assertEqual(result["logits"].argmax(1).item(), 0)
        torch.testing.assert_close(result["xy"], torch.tensor([[1., 1.]]))
        invalid = copy.deepcopy(payload)
        invalid["settings"]["neural_coordinate_weight"] = 0
        with self.assertRaisesRegex(ValueError, "positive neural"):
            RefinedKANMixer(Backbone(), invalid)
        invalid = copy.deepcopy(payload)
        invalid["fit_scope"] = "test"
        with self.assertRaisesRegex(ValueError, "train-only"):
            RefinedKANMixer(Backbone(), invalid)

    def test_ensemble_and_hybrid_checkpoint_are_self_contained_and_repeatable(self):
        config, data = self.fitted_config()
        members = [build_model(copy.deepcopy(config)).eval() for _ in range(2)]
        ensemble = EnsembleReferenceKANMixer(members).eval()
        x = torch.rand(2, 520, 2)
        x[..., 1] = (x[..., 1] > .5).float()
        with torch.inference_mode():
            outputs = [m(x) for m in members]
            result = ensemble(x)
        expected_prob = torch.stack([o["logits"].softmax(1) for o in outputs]).mean(0)
        torch.testing.assert_close(result["logits"].softmax(1), expected_prob)
        catalogue_x = torch.rand(26, 520, 2)
        catalogue_x[..., 1] = 1
        refinement = {"fit_scope": "internal_train_only", "signal": db_features(catalogue_x),
            "xy": data.xy, "class_ids": data.classes,
            "settings": {"neighbors": 3, "distance_power": 2., "neighbor_class_weight": .5,
                         "neural_coordinate_weight": .5}}
        wrapped = RefinedKANMixer(ensemble, refinement).eval()
        payload = {"format_version": 2, "config": config, "classes": [{}] * 13,
            "ensemble_members": [{"config": config, "model_state_dict": m.state_dict()} for m in members],
            "ensemble_confidence_weighted": True, "fingerprint_refinement": refinement}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ensemble.pt"
            torch.save(payload, path)
            loaded, _ = load_checkpoint(path, torch.device("cpu"))
            with torch.inference_mode():
                original, restored = wrapped(x), loaded(x)
            torch.testing.assert_close(original["xy"], restored["xy"], rtol=0, atol=0)
            torch.testing.assert_close(original["logits"], restored["logits"], rtol=0, atol=0)

    def test_final_refit_uses_frozen_epochs_and_blocks_in_sample_validation(self):
        from ips.algorithms.kan.final_refit import refit
        from ips.algorithms.kan.predict import evaluate
        config, train = self.fitted_config()
        classes = json.loads((ROOT / "data/processed/uji_group_split_v1/class_mapping.json").read_text())
        metadata = np.zeros((26, 9), dtype=np.float64)
        metadata[:, :2] = train.xy_original
        for row, label in enumerate(train.classes.tolist()):
            metadata[row, 2] = classes[label]["floor"]
            metadata[row, 3] = classes[label]["building"]
        development = SimpleNamespace(size=26, rssi=np.full((26, 520), -52, dtype=np.int16),
            metadata=metadata, pairs=[(c["building"], c["floor"]) for c in classes for _ in range(2)],
            column=lambda name: metadata[:, META_COLUMNS.index(name)])
        preprocessing = {"coordinates": {"center": [0., 0.], "shared_scale": 100.}}
        bundle = SimpleNamespace(directory=ROOT / "data/processed/uji_group_split_v1",
            development=development, preprocessing=preprocessing, classes=classes,
            ap_order=[f"WAP{i:03d}" for i in range(1, 521)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pt"
            torch.save({"format_version": 1, "config": config, "classes": classes,
                "model_state_dict": build_model(config).state_dict(), "metadata": {"config": config}}, source)
            lock = root / "selection.json"
            lock.write_text(json.dumps({"selection_split": "internal_validation_only", "test_evaluated": False,
                "final_refit_epochs": 1, "allowed_source_checkpoint_sha256": [sha256(source)]}))
            with patch("ips.algorithms.kan.final_refit.load_bundle", return_value=bundle):
                result = refit(source, lock, root / "refit", 42, "cpu")
            self.assertEqual(result["train_samples"], 26)
            self.assertEqual(result["epochs_completed"], 1)
            model, payload = load_checkpoint(root / "refit/best.pt", torch.device("cpu"))
            self.assertIsNone(payload["validation_mean_2d"])
            self.assertTrue(payload["metadata"]["full_development_refit"])
            self.assertEqual(payload["config"]["positioning"]["fit_scope"], "development_after_validation_selection")
            with patch("ips.algorithms.kan.predict.load_bundle", return_value=bundle):
                with self.assertRaisesRegex(ValueError, "incorporated"):
                    evaluate(root / "refit/best.pt", bundle.directory, "validation", root / "invalid_validation", "cpu")
