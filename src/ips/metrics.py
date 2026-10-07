"""Chỉ số sau inverse tọa độ; không dùng chúng để chọn theo test."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def inverse_xy(normalized, preprocessing):
    settings = preprocessing["coordinates"]
    return np.asarray(normalized, dtype=np.float64) * settings["shared_scale"] + np.asarray(settings["center"])


def evaluate_predictions(true_class, predicted_class, true_xy, predicted_xy, classes):
    true_class = np.asarray(true_class, dtype=np.int64)
    predicted_class = np.asarray(predicted_class, dtype=np.int64)
    true_xy, predicted_xy = np.asarray(true_xy, dtype=np.float64), np.asarray(predicted_xy, dtype=np.float64)
    n = len(true_class)
    if (n == 0 or predicted_class.shape != (n,) or true_xy.shape != (n, 2) or predicted_xy.shape != (n, 2)):
        raise ValueError("Metrics require nonempty aligned class[N] and xy[N,2] arrays.")
    if (not np.isfinite(true_xy).all() or not np.isfinite(predicted_xy).all() or
        (true_class < 0).any() or (predicted_class < 0).any() or
        (true_class >= len(classes)).any() or (predicted_class >= len(classes)).any()):
        raise ValueError("Invalid labels or non-finite coordinate predictions.")
    pairs = np.asarray([[c["building"], c["floor"]] for c in classes])
    true_pairs, predicted_pairs = pairs[true_class], pairs[predicted_class]
    delta = np.asarray(predicted_xy, dtype=np.float64) - np.asarray(true_xy, dtype=np.float64)
    errors = np.linalg.norm(delta, axis=1)
    correct_joint = predicted_class == true_class
    metrics = {
        "n_samples": len(errors),
        "joint_accuracy": float(accuracy_score(true_class, predicted_class)),
        "joint_macro_f1": float(f1_score(true_class, predicted_class, labels=list(range(len(classes))),
                                         average="macro", zero_division=0)),
        "joint_confusion_matrix": confusion_matrix(true_class, predicted_class,
                                                     labels=list(range(len(classes)))).tolist(),
        "mae_xy": np.abs(delta).mean(axis=0).tolist(),
        "rmse_xy": np.sqrt(np.square(delta).mean(axis=0)).tolist(),
        "mae_coordinates": float(np.abs(delta).mean()),
        "rmse_coordinates": float(np.sqrt(np.square(delta).mean())),
        "mean_2d": float(errors.mean()), "median_2d": float(np.median(errors)),
        "p90_2d": float(np.quantile(errors, 0.9)), "rmse_2d": float(np.sqrt(np.square(errors).mean())),
        "conditional_mean_2d_joint_correct": float(errors[correct_joint].mean()) if correct_joint.any() else None,
        "conditional_count_joint_correct": int(correct_joint.sum()),
        "coordinate_units": "UJIIndoorLoc original-coordinate metre convention; CRS not fully verified",
    }
    for axis, name in [(0, "building"), (1, "floor")]:
        labels = sorted(set(pairs[:, axis].tolist()))
        metrics[name + "_accuracy"] = float(accuracy_score(true_pairs[:, axis], predicted_pairs[:, axis]))
        metrics[name + "_macro_f1"] = float(f1_score(true_pairs[:, axis], predicted_pairs[:, axis],
                                                      labels=labels, average="macro", zero_division=0))
        metrics[name + "_confusion_matrix"] = confusion_matrix(true_pairs[:, axis], predicted_pairs[:, axis],
                                                                  labels=labels).tolist()
    metrics["per_building_floor"] = {
        f"{b}_{f}": {"count": int((true_class == i).sum()),
                      "joint_accuracy": float(correct_joint[true_class == i].mean()) if (true_class == i).any() else None,
                      "mean_2d": float(errors[true_class == i].mean()) if (true_class == i).any() else None}
        for i, (b, f) in enumerate(pairs)}
    metrics["thresholds"] = {str(threshold): {
        "fraction_2d_within": float((errors <= threshold).mean()),
        "fraction_joint_correct_and_2d_within": float(((errors <= threshold) & correct_joint).mean())}
        for threshold in [7, 8, 10, 20]}
    metrics["target_8m_met"] = bool(metrics["mean_2d"] <= 8)
    for axis, name in [(0, "building"), (1, "floor")]:
        metrics["per_" + name] = {str(int(label)): {
            "count": int((true_pairs[:, axis] == label).sum()),
            "mean_2d": float(errors[true_pairs[:, axis] == label].mean()) if (true_pairs[:, axis] == label).any() else None,
            "joint_accuracy": float(correct_joint[true_pairs[:, axis] == label].mean()) if (true_pairs[:, axis] == label).any() else None}
            for label in sorted(set(pairs[:, axis]))}
    metrics["error_by_classification_outcome"] = {}
    for name, mask in [("joint_correct", correct_joint), ("joint_wrong", ~correct_joint),
                       ("building_wrong", true_pairs[:, 0] != predicted_pairs[:, 0]),
                       ("floor_wrong", true_pairs[:, 1] != predicted_pairs[:, 1])]:
        metrics["error_by_classification_outcome"][name] = {
            "count": int(mask.sum()), "mean_2d": float(errors[mask].mean()) if mask.any() else None}
    return metrics
