"""Write per-run artifacts without modifying historical reports."""
import csv
from pathlib import Path

import numpy as np

from ..data import json_bytes
from .contracts import RunResult


def save_json(path: Path, payload):
    path.write_bytes(json_bytes(payload))


class ReportExporter:
    def export(self, result: RunResult, splits, directory: Path, plots: bool = True) -> dict[str, Path]:
        artifacts = {}
        for name, prediction in result.predictions.items():
            folder = directory/name
            folder.mkdir()
            path = folder/"predictions.csv"
            fields = ["source_id", "row_index", "csv_line_number"]
            output_names = {"class_ids": ["class_id"], "buildings": ["pred_building"],
                            "floors": ["pred_floor"], "confidence": ["confidence"],
                            "xy": ["pred_x", "pred_y"]}
            fields += [column for key, columns in output_names.items()
                       if getattr(prediction, key) is not None for column in columns]
            fields += ["true_class_id", "true_building", "true_floor", "true_x", "true_y"]
            truth = splits[name]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for i, row_index in enumerate(prediction.row_indices):
                    label = truth.info.classes[int(truth.class_ids[i])]
                    row = {"source_id": prediction.source_id, "row_index": int(row_index),
                           "csv_line_number": int(row_index) + 2, "true_class_id": int(truth.class_ids[i]),
                           "true_building": label["building"], "true_floor": label["floor"],
                           "true_x": float(truth.xy[i, 0]), "true_y": float(truth.xy[i, 1])}
                    for key, columns in output_names.items():
                        value = getattr(prediction, key)
                        if value is not None:
                            values = value[i] if key == "xy" else [value[i]]
                            row.update({column: v.item() for column, v in zip(columns, values)})
                    writer.writerow(row)
            artifacts[name + "_predictions"] = path
            for kind, payload in (("metrics", result.metrics[name]), ("matrices", result.matrices[name])):
                path = folder/(kind + ".json")
                save_json(path, payload)
                artifacts[name + "_" + kind] = path
            for matrix_name, matrix in result.matrices[name].items():
                path = folder/(matrix_name + ".csv")
                with path.open("w", encoding="utf-8", newline="") as stream:
                    labels = (list(range(len(truth.info.classes))) if matrix_name.startswith("joint") else
                              sorted({c["building" if matrix_name.startswith("building") else "floor"]
                                      for c in truth.info.classes}))
                    writer = csv.writer(stream)
                    writer.writerow(["true/predicted", *labels])
                    writer.writerows([[label, *row] for label, row in zip(labels, matrix)])
                artifacts[name + "_" + matrix_name] = path
            if plots:
                artifacts.update(self._plots(name, prediction, truth, result.matrices[name], folder))
        for name, payload in (("experiment_config", result.config_snapshot), ("history", result.history),
                              ("resources", result.resources)):
            if payload is not None:
                path = directory/(name + ".json")
                save_json(path, payload)
                artifacts[name] = path
        if result.diagram is not None:
            suffix = ".mmd" if result.diagram.format == "mermaid" else ".txt"
            path = directory/("model_diagram" + suffix)
            path.write_text(result.diagram.content, encoding="utf-8")
            artifacts["model_diagram"] = path
        return artifacts

    @staticmethod
    def _plots(name, prediction, truth, matrices, folder):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        artifacts = {}
        if prediction.xy is not None:
            errors = np.sort(np.linalg.norm(prediction.xy - truth.xy, axis=1))
            figure = Figure(figsize=(6, 4), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots()
            axes.plot(errors, np.arange(1, len(errors)+1)/len(errors))
            axes.set(xlabel="2D error (m)", ylabel="Cumulative fraction", title=name)
            axes.grid(alpha=.3)
            path = folder/"error_cdf.png"
            figure.savefig(path, dpi=150)
            artifacts[name + "_error_cdf"] = path
        for matrix_name, matrix in matrices.items():
            figure = Figure(figsize=(6, 5), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots()
            axes.imshow(matrix, cmap="Blues")
            labels = (list(range(len(truth.info.classes))) if matrix_name.startswith("joint") else
                      sorted({c["building" if matrix_name.startswith("building") else "floor"]
                              for c in truth.info.classes}))
            axes.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
                     yticklabels=labels, xlabel="Predicted", ylabel="True", title=name + " " + matrix_name)
            path = folder/(matrix_name + ".png")
            figure.savefig(path, dpi=150)
            artifacts[name + "_" + matrix_name + "_plot"] = path
        return artifacts
