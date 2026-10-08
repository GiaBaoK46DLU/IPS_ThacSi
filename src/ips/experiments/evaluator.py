"""Evaluate only outputs an algorithm actually supplies, using legacy IPS definitions."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ..data_access import FingerprintSplit
from ..metrics import evaluate_predictions
from .contracts import IPSPredictions


class IPSEvaluator:
    def evaluate(self, truth: FingerprintSplit, prediction: IPSPredictions, task: str = "auto"):
        prediction.validate(truth.inputs, truth.info.classes)
        classified = any(getattr(prediction, n) is not None for n in ("class_ids", "buildings", "floors"))
        if task == "position" and (prediction.xy is None or prediction.class_ids is None):
            raise ValueError("position requires coordinates and joint building/floor predictions.")
        if task == "coordinates" and prediction.xy is None:
            raise ValueError("coordinates requires xy predictions.")
        if task == "classification" and not classified:
            raise ValueError("classification requires building, floor or joint class predictions.")
        if prediction.xy is not None and prediction.class_ids is not None:
            metrics = evaluate_predictions(truth.class_ids, prediction.class_ids, truth.xy,
                                           prediction.xy, truth.info.classes)
        else:
            metrics = {"n_samples": len(truth.inputs)}
            pairs = np.asarray([(c["building"], c["floor"]) for c in truth.info.classes])
            if prediction.class_ids is not None:
                self._classification(metrics, "joint", truth.class_ids, prediction.class_ids,
                                     list(range(len(pairs))))
            for axis, name in enumerate(("building", "floor")):
                output = getattr(prediction, name + "s")
                if output is not None:
                    self._classification(metrics, name, pairs[truth.class_ids, axis], output,
                                         sorted(set(pairs[:, axis].tolist())))
            if prediction.xy is not None:
                delta = prediction.xy - truth.xy
                errors = np.linalg.norm(delta, axis=1)
                metrics.update({"mae_xy": np.abs(delta).mean(axis=0).tolist(),
                    "rmse_xy": np.sqrt(np.square(delta).mean(axis=0)).tolist(),
                    "mae_coordinates": float(np.abs(delta).mean()),
                    "rmse_coordinates": float(np.sqrt(np.square(delta).mean())),
                    "mean_2d": float(errors.mean()), "median_2d": float(np.median(errors)),
                    "p90_2d": float(np.quantile(errors, .9)), "rmse_2d": float(np.sqrt(np.square(errors).mean())),
                    "coordinate_units": "UJIIndoorLoc original-coordinate metre convention; CRS not fully verified",
                    "thresholds": {str(t): {"fraction_2d_within": float((errors <= t).mean())}
                                   for t in (7, 8, 10, 20)},
                    "target_8m_met": bool(errors.mean() <= 8)})
        matrices = {name: value for name, value in metrics.items() if name.endswith("_confusion_matrix")}
        return metrics, matrices

    @staticmethod
    def _classification(metrics, name, truth, prediction, labels):
        metrics[name + "_accuracy"] = float(accuracy_score(truth, prediction))
        metrics[name + "_macro_f1"] = float(f1_score(truth, prediction, labels=labels,
                                                    average="macro", zero_division=0))
        metrics[name + "_confusion_matrix"] = confusion_matrix(truth, prediction, labels=labels).tolist()
