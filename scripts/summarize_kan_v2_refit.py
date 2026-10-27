"""Publish audited V2 refit receipts after all ten fresh models finish."""
import argparse
import csv
import hashlib
import json
import shutil
import math
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    summary = read(run / "official_test_summary.json")
    protocol = read(run / "protocol.json")
    selection = read(run / "selection_lock.json")
    assert read(run / "state.json")["stage"] == "completed"
    assert summary["n_models"] == 10 and summary["test_rows_per_model"] == 1111
    assert protocol["historical_validation_seeds_reused"] == []
    assert protocol["all_ten_validation_trials_fresh"] is True
    assert all(r["reused_historical_validation"] is False for r in selection["all_ten_validation_results"])
    audits = [read(run / "official_test" / f"seed{s}" / "independent_audit.json") for s in summary["seeds"]]
    assert all(a["status"] == "passed" and a["rows"] == 1111 for a in audits)
    weight_audits = []
    for seed in summary["seeds"]:
        checkpoint = run / "final_refit" / f"seed{seed}" / "best.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        assert payload["seed"] == seed and payload["epoch"] == summary["refit_epochs"]
        assert metadata["full_development_refit"] is True
        assert metadata["actual_counts"] == {"train": 19937, "validation": 0}
        assert metadata["test_evaluated_during_training"] is False
        assert metadata["rng_reset_after_source_loading"] is True
        assert metadata["selection_lock_sha256"] == hashlib.sha256((run / "selection_lock.json").read_bytes()).hexdigest()
        original = metadata["source_training_config"]
        original["positioning"].pop("references", None)
        original["positioning"].pop("fit_scope", None)
        assert original == protocol["config"]
        training = read(checkpoint.parent / "summary.json")
        with (checkpoint.parent / "history.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        steps = int(history[-1]["optimizer_steps"])
        assert len(history) == training["epochs_completed"] == summary["refit_epochs"]
        assert steps == math.ceil(19937 / protocol["config"]["training"]["batch_size"]) * summary["refit_epochs"]
        digest = hashlib.sha256()
        for key, tensor in sorted(payload["model_state_dict"].items()):
            assert torch.isfinite(tensor).all()
            digest.update(key.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        weight_audits.append({"seed": seed, "state_dict_sha256": digest.hexdigest(),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "train_rows": 19937, "epochs": payload["epoch"], "optimizer_steps": steps})
    assert len({row["state_dict_sha256"] for row in weight_audits}) == 10
    public = ROOT / "docs/Bao/KAN_V2_Refit_10Seeds"
    public.mkdir(parents=True, exist_ok=True)
    receipts = ["protocol.json", "selection_lock.json", "test_checkpoint_lock.json", "official_test_summary.json"]
    receipts += [n for n in ("docs_relocation_amendment.json",) if (run / n).exists()]
    receipts += [p.name for p in sorted(run.glob("recovery_*.json"))]
    published = {}
    for name in receipts:
        source, destination = run / name, public / name
        shutil.copyfile(source, destination)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        published[name] = digest
    (public / "audit_summary.json").write_text(json.dumps({"status": "passed", "n_models": 10,
        "total_test_predictions": 11110, "all_validation_models_freshly_trained": True,
        "all_refits_freshly_initialized": True, "test_audits": audits,
        "distinct_state_dicts_verified": True, "weight_audits": weight_audits,
        "receipt_sha256": published}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    v3 = read(ROOT / "runs/kan_joint_under6_20261020_v1/official_test_summary.json")
    v3_group = next(g for g in v3["groups"] if g["role"] == "primary" and g["regime"] == "full_development_refit")
    assert v3_group["n_models"] == 10
    v3 = {"aggregate": {"mean_2d": {"mean": v3_group["mean_2d_mean"], "sample_std": v3_group["mean_2d_sample_std"]}}}
    v4 = read(ROOT / "docs/Bao/KAN_V4/official_test_summary.json")
    lines = ["# Kết quả KAN-Mixer V2 refit — 10 lượt train mới", "",
        f"Refit cố định **{summary['refit_epochs']} epoch** trên 19.937 dòng development, chọn từ đủ 10 best epoch Validation.", "",
        f"Best epoch theo thứ tự seed đã khai báo: {[r['best_epoch'] for r in selection['all_ten_validation_results']]}. "
        f"Median = {statistics.median(r['best_epoch'] for r in selection['all_ten_validation_results']):g}; "
        "làm tròn bằng `round()` của Python (trường hợp nửa đơn vị chọn số chẵn gần nhất), theo quy tắc khóa trước Test.", "",
        f"Mean2D Validation trước refit: **{statistics.mean(r['mean_2d'] for r in selection['all_ten_validation_results']):.6f} "
        f"± {statistics.stdev(r['mean_2d'] for r in selection['all_ten_validation_results']):.6f} m** (sample SD).", "",
        "| Phiên bản | Mean2D Test ± sample SD (m) | Số model |", "|---|---:|---:|"]
    for label, item in (("V2 refit mới", summary), ("V3", v3), ("V4", v4)):
        stats = item["aggregate"]["mean_2d"]
        lines.append(f"| {label} | {stats['mean']:.6f} ± {stats['sample_std']:.6f} | 10 |")
    lines += ["", "| Seed | Best epoch Validation | k | Temperature | Mean2D Test (m) |",
              "|---:|---:|---:|---:|---:|"]
    for trial, test in zip(selection["all_ten_validation_results"], summary["per_seed"]):
        assert trial["seed"] == test["seed"]
        decoder = trial["decoder"]
        lines.append(f"| {trial['seed']} | {trial['best_epoch']} | {decoder['top_k']} | {decoder['temperature']:g} | {test['mean_2d']:.6f} |")
    lines += ["", "| Chỉ số V2 refit | Trung bình | Sample SD giữa 10 model |",
              "|---|---:|---:|"]
    for key, label in (("median_2d", "Median (m)"), ("p90_2d", "P90 (m)"), ("rmse_2d", "RMSE (m)"),
                       ("building_accuracy", "Đúng tòa (%)"), ("floor_accuracy", "Đúng tầng (%)"), ("joint_accuracy", "Đúng tòa–tầng (%)")):
        stats = summary["aggregate"][key]
        scale = 100 if "accuracy" in key else 1
        lines.append(f"| {label} | {stats['mean']*scale:.6f} | {stats['sample_std']*scale:.6f} |")
    lines += ["", "Mỗi model được đánh giá trên đủ 1.111 dòng Test gốc; các metric được tính lại từ CSV dự đoán.",
        "Cả 10 lượt Validation đều train mới; không dùng ba checkpoint V2 lịch sử. Refit cũng khởi tạo trọng số mới.",
        "Con số V2 lịch sử 8,058 ± 0,306 m thuộc ba model train nội bộ; bảng này thuộc refit development.",
        "V3/V4 có cấu hình tăng cường dữ liệu và số epoch refit khác: đây là đối chiếu toàn bộ pipeline, chưa cô lập riêng một thay đổi.",
        "Test đã được xem trong lịch sử dự án; không chọn seed, decoder hoặc checkpoint sau khi xem Test của lượt này.", "",
        "[Quy trình và cấu hình](README.md) · [Kết quả JSON](official_test_summary.json) · [Kiểm toán](audit_summary.json)", ""]
    (public / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(public / "RESULTS.md")


if __name__ == "__main__":
    main()
