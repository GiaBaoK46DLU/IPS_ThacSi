"""Verify exported building experiments and create measured CSV/PNG/SVG tables.

DOCX authoring is a separate local helper; no fit or model selection here.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ...data import ROOT, sha256
from ...metrics import evaluate_predictions
from .runtime import save_json
from .uji_buildings import BUILDINGS, SEEDS, load_prepared, read

METHODS = ["joint_original", "joint_known_building", "separate_known_building"]
LABELS = {"joint_original": "Mạng chung", "joint_known_building": "Mạng chung biết tòa",
          "separate_known_building": "Mạng riêng từng tòa"}
COLORS = {"joint_original": "#4779a8", "joint_known_building": "#d49c44", "separate_known_building": "#228578"}
METRICS = ["mean_2d", "median_2d", "p90_2d", "rmse_2d", "floor_accuracy", "floor_macro_f1", "joint_accuracy"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_save(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "n": len(values)}


def figure(fig, destination, name):
    fig.savefig(destination / f"{name}.png", bbox_inches="tight")
    fig.savefig(destination / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def generate(output):
    assert read(output / "completed.json")["status"] == "completed"
    parent, parts = load_prepared(output)
    lock = read(output / "pre_test_lock.json")
    verification = []
    for entry in lock["experts"]:
        path = ROOT / entry["path"]
        assert sha256(path) == entry["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        building, seed = entry["building"], entry["seed"]
        info, datasets = parts[building]
        best = torch.load(output / f"building{building}/seed{seed}/training/best.pt", map_location="cpu", weights_only=True)
        assert all(torch.equal(value, best["model_state_dict"][key]) for key, value in payload["model_state_dict"].items())
        references = payload["config"]["positioning"]["references"]
        actual = {(r["class_id"], *r["xy"]) for r in references}
        expected = {(int(c), *xy) for c, xy in zip(datasets["train"].classes, datasets["train"].xy.tolist())}
        assert actual == expected
        assert all(c["building"] == building for c in payload["classes"])
        assert "fingerprint_refinement" not in payload
        assert payload["preprocessing"] == info.preprocessing
        assert payload["metadata"]["test_evaluated_during_training"] is False
        assert payload["split_manifest_sha256"] == sha256(info.directory / "manifest.json")
        for key, value in lock["joint_checkpoints"][str(seed)]["decoder"].items():
            assert payload["config"]["positioning"][key] == value
        verification.append({"seed": seed, "building": building, "checkpoint_sha256": entry["sha256"],
            "catalogue_train_only": True, "weights_equal_validation_selected_best": True,
            "references": len(references), "preprocessing_train_only": True})
    for setting in lock["joint_checkpoints"].values():
        assert sha256(ROOT / setting["joint_checkpoint"]) == setting["joint_checkpoint_sha256"]
    report = output / "report"
    tables, figures = report / "tables", report / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    raw = {name: parent.split(name) for name in ("train", "validation", "test")}
    history_path = ROOT / "runs/kan_seed44_complete_20261009_v1/report/per_seed_results.json"
    historical = {(r['seed'], r['split']): r['metrics'] for r in read(history_path) if r['variant'] == 'pure'}
    records, seed_records, prediction_cache = [], [], {}
    for method in METHODS:
        for split in raw:
            for seed in SEEDS:
                directory = output / f"evaluation/seed{seed}/{method}/{split}"
                saved = read(directory / "metrics.json")
                data = rows(directory / "predictions.csv")
                prediction_cache[method, split, seed] = data
                indices = np.asarray([int(r["row_index"]) for r in data])
                truth = np.asarray([[float(r["true_x"]), float(r["true_y"])] for r in data])
                xy = np.asarray([[float(r["predicted_x"]), float(r["predicted_y"])] for r in data])
                predicted = np.asarray([int(r["predicted_class"]) for r in data])
                errors = np.linalg.norm(xy-truth, axis=1)
                np.testing.assert_array_equal(indices, raw[split].inputs.row_indices)
                np.testing.assert_array_equal(truth, raw[split].xy)
                np.testing.assert_array_equal([int(r["true_class"]) for r in data], raw[split].class_ids)
                np.testing.assert_allclose(errors, [float(r["error_2d"]) for r in data], rtol=0, atol=1e-12)
                for key in ("USERID", "PHONEID"):
                    np.testing.assert_array_equal([int(r[key]) for r in data], raw[split].metadata[key])
                calculated = evaluate_predictions(raw[split].class_ids, predicted, truth, xy, parent.classes)
                for key in METRICS:
                    np.testing.assert_allclose(calculated[key], saved[key], rtol=0, atol=1e-12)
                for task in ("joint", "building", "floor"):
                    np.testing.assert_array_equal(calculated[f"{task}_confusion_matrix"], saved[f"{task}_confusion_matrix"])
                true_buildings = np.asarray([int(r["true_building"]) for r in data])
                means, counts = [], []
                for building in BUILDINGS:
                    mask = true_buildings == building
                    m = evaluate_predictions(raw[split].class_ids[mask], predicted[mask], truth[mask], xy[mask], parent.classes)
                    for key in METRICS:
                        np.testing.assert_allclose(m[key], saved["per_building"][str(building)][key], rtol=0, atol=1e-12)
                    means.append(m["mean_2d"])
                    counts.append(m["n_samples"])
                    records.append({"method": method, "split": split, "seed": seed, "building": building,
                                    "n": m["n_samples"], **{key: m[key] for key in METRICS}})
                np.testing.assert_allclose(np.mean(means), saved["macro_building_mean_2d"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(np.average(means, weights=counts), saved["mean_2d"], rtol=0, atol=1e-12)
                seed_records.append({"method": method, "split": split, "seed": seed,
                    "macro_building_mean_2d": saved["macro_building_mean_2d"],
                    **{key: saved[key] for key in METRICS}})
                if method == "joint_original":
                    np.testing.assert_allclose(saved["mean_2d"], historical[seed, split]["mean_2d"], rtol=0, atol=0.002)
    aggregate = {}
    for method in METHODS:
        aggregate[method] = {}
        for split in raw:
            sample = [r for r in seed_records if r["method"] == method and r["split"] == split]
            stats = {key: distribution([r[key] for r in sample]) for key in ["macro_building_mean_2d", *METRICS]}
            stats["buildings"] = {}
            for building in BUILDINGS:
                selected = [r for r in records if r["method"] == method and r["split"] == split and r["building"] == building]
                stats["buildings"][str(building)] = {key: distribution([r[key] for r in selected]) for key in METRICS}
                stats["buildings"][str(building)]["n"] = selected[0]["n"]
            aggregate[method][split] = stats
    resource_rows = []
    for building in BUILDINGS:
        for seed in SEEDS:
            directory = output / f"building{building}/seed{seed}/training"
            summary, resources = read(directory / "summary.json"), read(directory / "resources.json")
            resource_rows.append({"building": building, "seed": seed,
                "epochs": summary["epochs_completed"], "best_epoch": summary["best_epoch"],
                "train_seconds": summary["training_seconds"], "parameters": summary["trainable_parameters"],
                "RAM_peak_MiB": resources["training_phase"]["rss_peak_sampled_bytes"]/2**20,
                "VRAM_allocated_peak_MiB": resources["cuda_training_phase"]["peak_allocated_bytes"]/2**20,
                "VRAM_reserved_peak_MiB": resources["cuda_training_phase"]["peak_reserved_bytes"]/2**20})
    counts = []
    for building in BUILDINGS:
        info, _ = parts[building]
        counts.append({"building": building, "floors": len(info.classes),
            **{split: int(sum(parent.classes[int(c)]["building"] == building for c in raw[split].class_ids)) for split in raw},
            "coordinate_scale": info.preprocessing["coordinates"]["shared_scale"],
            "soft_sigma_metres": info.preprocessing["coordinates"]["shared_scale"]*.05,
            "references": next(v["references"] for v in verification if v["building"] == building)})
    csv_save(tables / "per_building_per_seed.csv", records)
    csv_save(tables / "macro_and_pooled_per_seed.csv", seed_records)
    csv_save(tables / "training_resources.csv", resource_rows)
    csv_save(tables / "dataset_counts.csv", counts)
    csv_save(tables / "test_building_summary.csv", [{"building": b, "n": aggregate[METHODS[0]]["test"]["buildings"][str(b)]["n"],
        **{f"{m}_{s}": aggregate[m]["test"]["buildings"][str(b)]["mean_2d"][s]
           for m in METHODS for s in ("mean", "std")}} for b in BUILDINGS])
    csv_save(tables / "test_overall_summary.csv", [{"method": m,
        **{f"{key}_{stat}": aggregate[m]["test"][key][stat] for key in ["macro_building_mean_2d", *METRICS] for stat in ("mean", "std")}}
        for m in METHODS])
    save_json(report / "aggregate.json", {"methods": aggregate, "counts": counts, "resources": resource_rows,
        "statistics": "mean and sample SD across 3 seeds; macro computed within seed before SD",
        "protocol": read(output / "protocol.json")})
    save_json(report / "verification.json", {"status": "passed", "models": verification,
        "predictions_source_rows_and_targets_verified": True, "metrics_recomputed_27_exports": True,
        "macro_and_pooled_formulas_verified": True, "joint_checkpoint_hashes_unchanged": True,
        "original_joint_scores_match_historical_within_2mm": True,
        "source_manifest_sha256": sha256(output / "protocol.json"),
        "limits": "prior test exposure, known-building routing, different coordinate scaling and physical sigma, seed-specific decoder, 3 seeds only"})

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    x = np.arange(3)
    for j, method in enumerate(METHODS):
        stats = [aggregate[method]["test"]["buildings"][str(b)]["mean_2d"] for b in BUILDINGS]
        bars = ax.bar(x+(j-1)*.24, [s["mean"] for s in stats], .23,
                      yerr=[s["std"] for s in stats], capsize=3, color=COLORS[method], label=LABELS[method])
        ax.bar_label(bars, fmt="%.2f", padding=7, fontsize=12)
    ax.set_xticks(x, [f"Tòa {b}" for b in BUILDINGS])
    ax.set_ylabel("Sai số 2D trung bình trên test (m)")
    ax.set_ylim(0, max(s["mean"]+s["std"] for m in METHODS for s in [aggregate[m]["test"]["buildings"][str(b)]["mean_2d"] for b in BUILDINGS])*1.28)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.22), frameon=False, fontsize=12)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    figure(fig, figures, "01_test_tung_toa")

    fig, ax = plt.subplots(figsize=(9.5, 4))
    for j, method in enumerate(METHODS):
        selected = [aggregate[method]["test"][key] for key in ("macro_building_mean_2d", "mean_2d")]
        bars = ax.bar(np.arange(2)+(j-1)*.24, [s["mean"] for s in selected], .23,
                      yerr=[s["std"] for s in selected], capsize=3, label=LABELS[method], color=COLORS[method])
        ax.bar_label(bars, fmt="%.3f", padding=6, fontsize=12)
    ax.set_xticks([0, 1], ["Trung bình đều 3 tòa", "Trung bình theo số mẫu"])
    ax.set_ylabel("Sai số 2D trên test (m)")
    ax.set_ylim(0, max(aggregate[m]["test"][key]["mean"]+aggregate[m]["test"][key]["std"]
                     for m in METHODS for key in ("macro_building_mean_2d", "mean_2d"))*1.25)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.22), frameon=False, fontsize=12)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    figure(fig, figures, "02_macro_va_theo_so_mau")

    fig, ax = plt.subplots(figsize=(9.5, 4))
    for j, (split, label, color) in enumerate([("train", "Train", "#648da4"), ("validation", "Validation", "#d49c44"), ("test", "Test", "#228578")]):
        selected = [aggregate["separate_known_building"][split]["buildings"][str(b)]["mean_2d"] for b in BUILDINGS]
        bars = ax.bar(x+(j-1)*.24, [s["mean"] for s in selected], .23,
                      yerr=[s["std"] for s in selected], capsize=3, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=6, fontsize=12)
    ax.set_xticks(x, [f"Tòa {b}" for b in BUILDINGS])
    ax.set_ylabel("Sai số 2D trung bình (m)")
    ax.margins(y=.20)
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    figure(fig, figures, "03_cac_tap_mang_rieng")

    fig, ax = plt.subplots(figsize=(9.5, 4))
    for building, color in zip(BUILDINGS, ["#4779a8", "#d49c44", "#228578"]):
        errors = np.sort([float(r["error_2d"]) for seed in SEEDS for r in prediction_cache["separate_known_building", "test", seed]
                          if int(r["true_building"]) == building])
        ax.plot(errors, np.arange(1,len(errors)+1)/len(errors), lw=2, color=color, label=f"Tòa {building}")
    ax.axvline(8, color="#7b8790", linestyle="--", lw=1, label="Mốc 8 m")
    ax.set(xlim=(0,30), ylim=(0,1.01), xlabel="Sai số 2D (m)", ylabel="Tỷ lệ tích lũy")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=.18)
    figure(fig, figures, "04_cdf_test_mang_rieng")

    fig, axes = plt.subplots(1,3,figsize=(10.5,3.5))
    for building, ax in zip(BUILDINGS, axes):
        labels = [c["floor"] for c in parent.classes if c["building"] == building]
        selected = [r for seed in SEEDS for r in prediction_cache["separate_known_building", "test", seed] if int(r["true_building"]) == building]
        matrix = np.zeros((len(labels),len(labels)), dtype=int)
        mapping = {v:i for i,v in enumerate(labels)}
        for r in selected:
            matrix[mapping[int(r["true_floor"])]][mapping[int(r["predicted_floor"])]] += 1
        fraction = matrix/np.maximum(matrix.sum(1,keepdims=True),1)
        ax.imshow(fraction,vmin=0,vmax=1,cmap="Blues")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=13,color="white" if fraction[i,j]>.55 else "#243b50")
        ax.set(xticks=np.arange(len(labels)),yticks=np.arange(len(labels)),xticklabels=labels,yticklabels=labels,
               xlabel="Tầng dự đoán",ylabel="Tầng thật",title=f"Tòa {building}")
        np.savetxt(tables/f"test_floor_confusion_building{building}_3seeds.csv",matrix,fmt="%d",delimiter=",")
    fig.tight_layout(w_pad=2)
    figure(fig,figures,"05_ma_tran_tang_test")

    fig, axes = plt.subplots(1,3,figsize=(10,3.8),sharey=True)
    history_max = 0.0
    for building, ax in zip(BUILDINGS, axes):
        for seed,color in zip(SEEDS,["#4779a8","#d49c44","#228578"]):
            history=rows(output/f"building{building}/seed{seed}/training/history.csv")
            history_max=max(history_max,max(float(r["val_mean_2d"]) for r in history))
            ax.plot([int(r["epoch"]) for r in history],[float(r["val_mean_2d"]) for r in history],color=color,label=f"Seed {seed}")
        ax.set(title=f"Tòa {building}",xlabel="Epoch")
        ax.grid(alpha=.16)
    axes[0].set_ylabel("Mean 2D validation (m)")
    axes[0].set_ylim(0,history_max*1.08)
    axes[-1].legend(frameon=False,fontsize=13)
    fig.tight_layout()
    figure(fig,figures,"06_validation_theo_epoch")
    print(json.dumps({"verification":"passed","test_summary":{
        m:{key:aggregate[m]["test"][key] for key in ["macro_building_mean_2d","mean_2d"]} for m in METHODS}},ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    generate(args.output_dir.resolve())


if __name__ == "__main__":
    main()
