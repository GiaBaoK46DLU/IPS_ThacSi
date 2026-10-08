"""Strict fairness checks and reports generated only from completed real runs."""
import copy
import csv
import json
import math
from pathlib import Path
import shutil

import numpy as np

from .data import ROOT, csv_bytes, sha256
from .dataset import load_bundle
from .metrics import evaluate_predictions
from .runtime import save_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Aggregation needs finite nonempty per-seed scores.')
    return {'n': len(values), 'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if len(values) > 1 else None}


def audit_suite(root):
    root = Path(root)
    protocol = read_json(root / 'protocol_lock.json')
    if protocol['models'] != ['mlp', 'mlp_mixer', 'kan_mixer'] or protocol['seeds'] != [42, 43, 44]:
        raise ValueError('Stage 7 requires exactly three declared models and seeds 42/43/44.')
    if protocol['device'] != 'cuda' or protocol['config']['training']['max_epochs'] != 100:
        raise ValueError('Wrong frozen device or epoch budget.')
    if read_json(root / 'frozen_config.json') != protocol['config']:
        raise ValueError('Frozen configuration changed.')
    for path, digest in protocol['training_implementation_sha256'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f'Training implementation changed since protocol lock: {path}')
    expected_names = {f'{model}_seed{seed}' for model in protocol['models'] for seed in protocol['seeds']}
    if {p.name for p in root.glob('*_seed*') if p.is_dir()} != expected_names:
        raise ValueError('Missing, extra or duplicate experiment runs.')
    items = []
    reference_software = None
    for model in protocol['models']:
        for seed in protocol['seeds']:
            directory = root / f'{model}_seed{seed}'
            metadata = read_json(directory / 'run_metadata.json')
            summary = read_json(directory / 'summary.json')
            resources = read_json(directory / 'resources.json')
            config = copy.deepcopy(protocol['config'])
            config['model'] = model
            if (metadata['run_type'] != 'training' or summary['run_type'] != 'training' or
                    summary['status'] != 'completed' or metadata['seed'] != seed or
                    summary['seed'] != seed or summary['model'] != model or metadata['config'] != config or
                    read_json(directory / 'resolved_config.json') != config or
                    metadata['resolved_max_epochs'] != 100 or
                    metadata['actual_counts'] != {'train': 16025, 'validation': 3912} or
                    metadata['split_manifest_sha256'] != protocol['split_manifest_sha256'] or
                    metadata['training_implementation_sha256'] != protocol['training_implementation_sha256'] or
                    metadata['test_evaluated_during_training'] or summary['test_evaluated']):
                raise ValueError(f'Fairness mismatch, incomplete training or smoke run: {directory.name}')
            software = metadata['software']
            if software['device'] != 'cuda' or resources['software'] != software:
                raise ValueError(f'Wrong device/resource provenance: {directory.name}')
            if reference_software is None:
                reference_software = software
            if software != reference_software:
                raise ValueError(f'Hardware/software/threads differ: {directory.name}')
            history = read_csv(directory / 'history.csv')
            n = summary['epochs_completed']
            if not 1 <= n <= 100 or [int(r['epoch']) for r in history] != list(range(1, n + 1)):
                raise ValueError(f'Invalid or truncated epoch history: {directory.name}')
            if summary['optimizer_steps'] != n * math.ceil(16025 / config['training']['batch_size']):
                raise ValueError(f'Wrong optimizer-step budget: {directory.name}')
            best, best_epoch = float('inf'), None
            for row in history:
                current = float(row['val_mean_2d'])
                if not math.isfinite(current) or float(row['epoch_seconds']) <= 0:
                    raise ValueError('Invalid measured values in history.')
                if current < best - config['training']['min_delta']:
                    best, best_epoch = current, int(row['epoch'])
            if (summary['best_epoch'] != best_epoch or abs(summary['best_validation_mean_2d'] - best) > 1e-10 or
                    abs(read_json(directory / 'best_validation_metrics.json')['mean_2d'] - best) > 1e-10):
                raise ValueError(f'Checkpoint selection disagrees with frozen rule: {directory.name}')
            if resources['training_phase_seconds'] != summary['training_seconds']:
                raise ValueError('Resource timing and training summary disagree.')
            items.append({'model': model, 'seed': seed, 'run': directory.name,
                          'checkpoint_sha256': sha256(directory / 'best.pt'),
                          'epochs_completed': n, 'best_epoch': best_epoch,
                          'optimizer_steps': summary['optimizer_steps']})
    return {'status': 'passed', 'models': protocol['models'], 'seeds': protocol['seeds'], 'runs': items,
            'split_manifest_sha256': protocol['split_manifest_sha256'], 'software': reference_software,
            'protocol': protocol,
            'limitations': ['Equal maximum epochs and stopping rule, not equal actual steps/FLOPs or parameter counts.',
                            'Proxy measurement groups; not a guarantee of unseen users, locations or devices.']}


def check_predictions(rows, dataset, classes):
    indices = np.asarray([int(row['row_index']) for row in rows])
    if not np.array_equal(indices, dataset.indices.numpy()):
        raise ValueError('Evaluation rows must match the saved split in original order.')
    true_xy = np.asarray([[float(r['true_x']), float(r['true_y'])] for r in rows])
    predicted_xy = np.asarray([[float(r['pred_x']), float(r['pred_y'])] for r in rows])
    mapping = {(c['building'], c['floor']): c['class_id'] for c in classes}
    true_classes = np.asarray([mapping[(int(r['true_building']), int(r['true_floor']))] for r in rows])
    predicted_classes = np.asarray([int(r['class_id']) for r in rows])
    if not np.array_equal(true_xy, dataset.xy_original) or not np.array_equal(true_classes, dataset.classes.numpy()):
        raise ValueError('Ground truth does not match saved dataset rows.')
    if any((int(r['pred_building']), int(r['pred_floor'])) !=
           (classes[int(r['class_id'])]['building'], classes[int(r['class_id'])]['floor']) for r in rows):
        raise ValueError('Predicted class/building/floor mapping mismatch.')
    errors = np.linalg.norm(predicted_xy - true_xy, axis=1)
    if not np.allclose(errors, [float(r['error_2d']) for r in rows], rtol=0, atol=1e-9):
        raise ValueError('Stored Euclidean errors disagree with coordinates.')
    if any(int(r['csv_line_number']) != int(r['row_index']) + 2 for r in rows):
        raise ValueError('CSV line numbers are misaligned.')
    for name in ['USERID', 'PHONEID']:
        if not np.array_equal([int(r[name]) for r in rows], dataset.metadata[name].astype(int)):
            raise ValueError('Wrong metadata group labels.')
    return evaluate_predictions(true_classes, predicted_classes, true_xy, predicted_xy, classes)


def aggregate_groups(model, split, metrics):
    rows = []
    sources = [('building', 'per_building'), ('floor', 'per_floor'),
               ('building_floor', 'per_building_floor'), ('detected_aps', 'per_detected_ap_bin')]
    sources += [('USERID', None), ('PHONEID', None)]
    for kind, key in sources:
        groups = [m[key] if key else m['per_metadata_group'][kind] for m in metrics]
        if any(set(g) != set(groups[0]) for g in groups):
            raise ValueError('Group coverage differs between seeds.')
        for label in groups[0]:
            values = [g[label] for g in groups]
            counts = {v['count'] for v in values}
            if len(counts) != 1:
                raise ValueError('Group supports differ between seeds.')
            count = counts.pop()
            mean = mean_std([v['mean_2d'] for v in values]) if count else None
            accuracy = mean_std([v['joint_accuracy'] for v in values]) if count else None
            rows.append([model, split, kind, label, count,
                         mean['mean'] if mean else None, mean['std'] if mean else None,
                         accuracy['mean'] if accuracy else None, accuracy['std'] if accuracy else None,
                         'unrecorded_user_not_a_person' if split == 'test' and kind == 'USERID' else ''])
    return rows


def fmt(stat, percentage=False):
    scale = 100 if percentage else 1
    return f"{stat['mean']*scale:.2f} ± {stat['std']*scale:.2f}" if stat['std'] is not None else f"{stat['mean']*scale:.2f} (n=1)"


def plot_results(output, records, classes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'mlp': '#2563eb', 'mlp_mixer': '#059669', 'kan_mixer': '#d97706'}
    plots = output / 'figures'
    plots.mkdir()
    def save(fig, name):
        fig.savefig(plots / (name + '.png'), dpi=160, bbox_inches='tight')
        fig.savefig(plots / (name + '.svg'), bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model, runs in records.items():
        for index, run in enumerate(runs):
            h = run['history']
            axes[0].plot([int(r['epoch']) for r in h], [float(r['val_mean_2d']) for r in h],
                         color=colors[model], alpha=.6, label=model if index == 0 else None)
            axes[1].plot([int(r['epoch']) for r in h], [float(r['train_loss']) for r in h], color=colors[model], alpha=.6)
    axes[0].set(xlabel='Epoch', ylabel='Validation mean 2D error (m)', title='Each line is one seed; internal validation')
    axes[1].set(xlabel='Epoch', ylabel='Combined training loss', title='CE + normalized-coordinate SmoothL1')
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, 'learning_curves')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    zoom_limit = 8.
    for model, runs in records.items():
        for index, run in enumerate(runs):
            errors = np.sort([float(r['error_2d']) for r in run['test_rows']])
            zoom_limit = max(zoom_limit, float(np.quantile(errors, .95)))
            for ax in axes:
                ax.step(errors, np.arange(1, len(errors)+1) / len(errors), where='post',
                        color=colors[model], alpha=.6, label=model if index == 0 else None)
    for ax, title in zip(axes, ['Zoomed axis (all test rows retained)', 'Full error range (all test rows)']):
        ax.axvline(8, color='black', linestyle='--', linewidth=1, label='8 m individual-error threshold')
        ax.set(xlabel='2D error (m)', ylabel='Fraction of all test rows', title=title)
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_xlim(0, 1.1 * zoom_limit)
    fig.tight_layout()
    save(fig, 'test_error_cdf')

    fig, ax = plt.subplots(figsize=(11, 4))
    locations = np.arange(len(classes))
    width = .25
    for i, (model, runs) in enumerate(records.items()):
        values = [[run['test']['per_building_floor'][f"{c['building']}_{c['floor']}"]['mean_2d']
                   for c in classes] for run in runs]
        values = np.asarray(values, dtype=float)
        ax.bar(locations+(i-1)*width, values.mean(axis=0), width, yerr=values.std(axis=0, ddof=1),
               color=colors[model], label=model, capsize=2)
    ax.set_xticks(locations, [f"{c['building']}/{c['floor']}" for c in classes])
    ax.set(xlabel='True building/floor', ylabel='Mean 2D error (m)', title='Test: mean ± sample SD across seeds')
    ax.legend()
    fig.tight_layout()
    save(fig, 'test_error_by_building_floor')

    for model, runs in records.items():
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for ax, key, labels in zip(axes, ['building', 'floor', 'joint'],
                                  [list(range(3)), list(range(5)), [f"{c['building']}/{c['floor']}" for c in classes]]):
            matrix = np.mean([run['test'][key+'_confusion_matrix'] for run in runs], axis=0)
            support = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(matrix, support, out=np.zeros_like(matrix), where=support != 0)
            ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
            ax.set_xticks(range(len(labels)), labels, rotation=90 if key == 'joint' else 0)
            ax.set_yticks(range(len(labels)), labels)
            ax.set(xlabel='Predicted', ylabel='True', title=key + ': row-normalized')
            for y in range(len(labels)):
                for x in range(len(labels)):
                    if normalized[y, x] > 0:
                        ax.text(x, y, f'{normalized[y,x]:.2f}', ha='center', va='center',
                                fontsize=7, color='white' if normalized[y,x] > .5 else 'black')
        fig.suptitle(f'{model}: mean confusion counts over 3 seeds; same 1111 test rows')
        fig.tight_layout()
        save(fig, f'confusion_{model}')


def report(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(f'Report directory already exists: {output}')
    audit = audit_suite(root)
    pretest = read_json(root / 'pretest_audit.json')
    if audit['runs'] != pretest['runs']:
        raise ValueError('Chosen checkpoints changed after pretest audit.')
    bundle = load_bundle(ROOT / audit['protocol']['config']['split_dir'])
    datasets = {split: bundle.dataset(split) for split in ['validation', 'test']}
    records, aggregates, groups, individual, worst = {}, {}, [], [], []
    scalar_keys = ['building_accuracy', 'building_macro_f1', 'floor_accuracy', 'floor_macro_f1',
                   'joint_accuracy', 'joint_macro_f1', 'mae_coordinates', 'rmse_coordinates',
                   'mean_2d', 'median_2d', 'p90_2d', 'rmse_2d']
    for model in audit['models']:
        records[model] = []
        for seed in audit['seeds']:
            directory = root / f'{model}_seed{seed}'
            entry = {'directory': directory, 'seed': seed, 'summary': read_json(directory / 'summary.json'),
                     'resources': read_json(directory / 'resources.json'), 'history': read_csv(directory / 'history.csv')}
            checkpoint_hash = sha256(directory / 'best.pt')
            for split in ['validation', 'test']:
                metrics = read_json(directory / split / 'metrics.json')
                rows = read_csv(directory / split / 'predictions.csv')
                if (metrics['checkpoint_sha256'] != checkpoint_hash or metrics['model'] != model or
                        metrics['seed'] != seed or metrics['split'] != split or metrics['run_type'] != 'training' or
                        metrics['split_manifest_sha256'] != audit['split_manifest_sha256']):
                    raise ValueError('Evaluation/checkpoint provenance mismatch.')
                computed = check_predictions(rows, datasets[split], bundle.classes)
                for key in scalar_keys:
                    if not np.isclose(metrics[key], computed[key], rtol=0, atol=1e-9):
                        raise ValueError('Reported score disagrees with saved predictions: ' + key)
                for key in ['joint_confusion_matrix', 'building_confusion_matrix', 'floor_confusion_matrix', 'thresholds']:
                    if metrics[key] != computed[key]:
                        raise ValueError('Evaluation matrix/threshold disagrees with predictions.')
                entry[split], entry[split+'_rows'] = metrics, rows
                individual.append([model, seed, split, *[metrics[k] for k in scalar_keys],
                                   *metrics['mae_xy'], *metrics['rmse_xy'], metrics['thresholds']['8']['fraction_2d_within'],
                                   metrics['thresholds']['8']['fraction_joint_correct_and_2d_within']])
                if split == 'test':
                    for row in sorted(rows, key=lambda r: float(r['error_2d']), reverse=True)[:20]:
                        worst.append([model, seed] + [row[k] for k in ['row_index', 'true_building', 'true_floor',
                            'pred_building', 'pred_floor', 'error_2d', 'PHONEID', 'detected_aps']])
            entry['benchmarks'] = {}
            for batch in audit['protocol']['benchmark']['batch_sizes']:
                bench = read_json(directory / 'benchmarks' / f'batch{batch}.json')
                if (bench['checkpoint_sha256'] != checkpoint_hash or bench['model'] != model or bench['seed'] != seed or
                        bench['batch_size'] != batch or bench['software'] != audit['software'] or
                        bench['warmup'] != audit['protocol']['benchmark']['warmup'] or
                        bench['repeats'] != audit['protocol']['benchmark']['repeats'] or
                        bench['input_sha256'] != sha256(root / 'benchmark_inputs.csv')):
                    raise ValueError('Benchmark protocol or checkpoint differs between runs.')
                entry['benchmarks'][batch] = bench
            records[model].append(entry)
        aggregates[model] = {}
        for split in ['validation', 'test']:
            metrics = [run[split] for run in records[model]]
            scores = {key: mean_std([m[key] for m in metrics]) for key in scalar_keys}
            for axis in range(2):
                for key in ['mae_xy', 'rmse_xy']:
                    scores[f'{key}_{axis}'] = mean_std([m[key][axis] for m in metrics])
            for threshold in ['7', '8', '10', '20']:
                for key in ['fraction_2d_within', 'fraction_joint_correct_and_2d_within']:
                    scores[f'{key}_{threshold}m'] = mean_std([m['thresholds'][threshold][key] for m in metrics])
            aggregates[model][split] = scores
            groups += aggregate_groups(model, split, metrics)
        runs = records[model]
        aggregates[model]['resources'] = {
            'training_seconds': mean_std([r['summary']['training_seconds'] for r in runs]),
            'epochs_completed': mean_std([r['summary']['epochs_completed'] for r in runs]),
            'optimizer_steps': mean_std([r['summary']['optimizer_steps'] for r in runs]),
            'train_ram_peak_mib': mean_std([r['resources']['training_phase']['rss_peak_sampled_bytes']/2**20 for r in runs]),
            'pipeline_ram_peak_mib': mean_std([r['resources']['pipeline']['rss_peak_sampled_bytes']/2**20 for r in runs]),
            'train_vram_allocated_mib': mean_std([r['resources']['cuda_training_phase']['peak_allocated_bytes']/2**20 for r in runs]),
            'train_vram_reserved_mib': mean_std([r['resources']['cuda_training_phase']['peak_reserved_bytes']/2**20 for r in runs]),
            'trainable_parameters': runs[0]['summary']['trainable_parameters'],
            'parameter_tensor_bytes': runs[0]['resources']['parameter_tensor_bytes']}
        aggregates[model]['benchmarks'] = {}
        for batch in [1, 64]:
            aggregates[model]['benchmarks'][str(batch)] = {mode: {
                key: mean_std([r['benchmarks'][batch]['modes'][mode][key] for r in runs])
                for key in ['median_ms', 'p90_ms', 'mean_ms', 'throughput_samples_per_second', 'amortized_mean_ms_per_sample']}
                for mode in ['forward', 'end_to_end']}
            for mode in ['forward', 'end_to_end']:
                aggregates[model]['benchmarks'][str(batch)][mode].update({
                    'ram_peak_mib': mean_std([r['benchmarks'][batch]['modes'][mode]['ram']['rss_peak_sampled_bytes']/2**20 for r in runs]),
                    'vram_allocated_mib': mean_std([r['benchmarks'][batch]['modes'][mode]['cuda']['peak_allocated_bytes']/2**20 for r in runs]),
                    'vram_reserved_mib': mean_std([r['benchmarks'][batch]['modes'][mode]['cuda']['peak_reserved_bytes']/2**20 for r in runs])})

    output.mkdir(parents=True)
    save_json(output / 'fairness_audit.json', audit)
    save_json(output / 'aggregate_metrics.json', aggregates)
    (output / 'per_seed_metrics.csv').write_bytes(csv_bytes(['model', 'seed', 'split']+scalar_keys+
        ['mae_x', 'mae_y', 'rmse_x', 'rmse_y', 'fraction_within_8m', 'fraction_joint_correct_within_8m'], individual))
    (output / 'group_metrics.csv').write_bytes(csv_bytes(['model', 'split', 'group_kind', 'group', 'count',
        'mean_2d_mean', 'mean_2d_std', 'joint_accuracy_mean', 'joint_accuracy_std', 'note'], groups))
    (output / 'worst_test_errors.csv').write_bytes(csv_bytes(['model', 'seed', 'row_index', 'true_building',
        'true_floor', 'pred_building', 'pred_floor', 'error_2d', 'PHONEID', 'detected_aps'], worst))
    diagnostic_rows = []
    for run in records['kan_mixer']:
        frames = read_json(run['directory'] / 'kan_diagnostics.json')
        for role, epoch in [('best', run['summary']['best_epoch']), ('last', run['summary']['epochs_completed'])]:
            frame = next(f for f in frames if f['epoch'] == epoch)
            for layer, values in frame['layers'].items():
                diagnostic_rows.append([run['seed'], role, epoch, frame['sampled_train_batches'], layer,
                    *[values[k] for k in ['n_values', 'mean', 'std', 'min', 'max',
                                         'outside_nominal_fraction', 'outside_knot_support_fraction']]])
    (output / 'kan_diagnostics_summary.csv').write_bytes(csv_bytes(['seed', 'checkpoint_role', 'epoch',
        'sampled_train_batches', 'layer', 'n_values', 'mean', 'std', 'min', 'max',
        'outside_nominal_fraction', 'outside_knot_support_fraction'], diagnostic_rows))
    for name in ['protocol_lock.json', 'pretest_audit.json', 'benchmark_inputs.csv', 'benchmark_input_provenance.json',
                 'frozen_config.json', 'terminal.log', 'evaluation_terminal.log', 'evaluation_provenance.json']:
        shutil.copyfile(root / name, output / name)
    artifacts = output / 'artifacts'
    for model, runs in records.items():
        for run in runs:
            source = run['directory']
            target = artifacts / source.name
            target.mkdir(parents=True)
            for name in ['summary.json', 'run_metadata.json', 'resources.json', 'history.csv', 'kan_diagnostics.json']:
                shutil.copyfile(source / name, target / name)
            for name in ['validation', 'test', 'benchmarks']:
                shutil.copytree(source / name, target / name)
    plot_results(output, records, bundle.classes)
    lines = ['# Kết quả chặng 7: UJIIndoorLoc', '',
        'Bảng được sinh từ log và dự đoán thật sau khi kiểm tra giao thức. Mỗi ô tổng hợp là trung bình ± độ lệch chuẩn mẫu trên seed42/43/44 (ddof=1), không phải khoảng tin cậy.', '',
        f"Thiết bị: {audit['software']['device_name']}; PyTorch{audit['software']['torch']}; batch64; FP32; 4 luồng CPU. Train16.025/validation3.912/test1.111. Tất cả checkpoint chọn bằng validation nội bộ; test giữ nguyên toàn bộ mẫu.", '']
    for split, title in [('validation', 'Validation nội bộ'), ('test', 'Test chính thức')]:
        lines += ['## '+title, '', '| Mô hình | Building acc (%) | Building macro-F1 (%) | Floor acc (%) | Floor macro-F1 (%) | Joint acc (%) | Joint macro-F1 (%) |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for model in records:
            scores = aggregates[model][split]
            keys = ['building_accuracy', 'building_macro_f1', 'floor_accuracy', 'floor_macro_f1', 'joint_accuracy', 'joint_macro_f1']
            lines.append('| '+model+' | '+' | '.join(fmt(scores[k], True) for k in keys)+' |')
        lines += ['', '| Mô hình | MAE tọa độ (m) | RMSE tọa độ (m) | Mean 2D (m) | Median 2D (m) | P90 2D (m) | RMSE 2D (m) |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for model in records:
            scores = aggregates[model][split]
            keys = ['mae_coordinates', 'rmse_coordinates', 'mean_2d', 'median_2d', 'p90_2d', 'rmse_2d']
            lines.append('| '+model+' | '+' | '.join(fmt(scores[k]) for k in keys)+' |')
        lines += ['', 'MAE/RMSE tọa độ lấy trên hai trục; mean2D là trung bình khoảng cách Euclid của từng mẫu. Không thay thế các chỉ số này cho nhau. MAE/RMSE từng trục có trong `per_seed_metrics.csv` và `aggregate_metrics.json`.', '']
    lines += ['## Kiểm chứng mục tiêu 7–8 m', '',
              '| Mô hình | Mean 2D test (m) | Seed đạt mean≤8 m | Mẫu có lỗi≤8 m (%) | Đúng building/floor và lỗi≤8 m (%) |',
              '|---|---:|---:|---:|---:|']
    for model, runs in records.items():
        scores = aggregates[model]['test']
        count = sum(r['test']['mean_2d'] <= 8 for r in runs)
        lines.append(f"| {model} | {fmt(scores['mean_2d'])} | {count}/3 | {fmt(scores['fraction_2d_within_8m'], True)} | {fmt(scores['fraction_joint_correct_and_2d_within_8m'], True)} |")
    met = [m for m in records if aggregates[m]['test']['mean_2d']['mean'] <= 8]
    lines += ['', ('Trung bình qua ba seed đạt ngưỡng8 m: '+', '.join(met)) if met else
              '**Chưa đạt mục tiêu mean2D≤8 m trên toàn bộ test với cấu hình đã khóa.** Đây là kết quả đo, không được sửa bảng hay tinh chỉnh theo test để đạt mục tiêu.', '',
              'Ngưỡng8 m cho từng mẫu là một chỉ số khác với mean≤8 m. Sai tòa/tầng không được loại khỏi mean2D; xem riêng lỗi có điều kiện trong metrics từng seed.', '']
    lines += ['## Chi phí huấn luyện', '', '| Mô hình | Tham số | Epoch thực chạy | Thời gian train+val (giây) | RAM peak train (MiB) | VRAM allocated (MiB) | VRAM reserved (MiB) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for model in records:
        r = aggregates[model]['resources']
        lines.append(f"| {model} | {r['trainable_parameters']} | {fmt(r['epochs_completed'])} | {fmt(r['training_seconds'])} | {fmt(r['train_ram_peak_mib'])} | {fmt(r['train_vram_allocated_mib'])} | {fmt(r['train_vram_reserved_mib'])} |")
    total = sum(run['summary']['training_seconds'] for runs in records.values() for run in runs)
    lines += ['', f'Tổng thời gian vòng huấn luyện của9 lượt: **{total/60:.2f} phút**. Có validation, diagnostics, logging và lưu checkpoint; không gồm khởi tạo/tải dữ liệu và công đoạn đánh giá cuối. Pipeline RAM peak được lưu riêng trong JSON.', '',
              'Cùng ngân sách tối đa100 epoch, patience15, min_delta0,01 m; số epoch/steps thực tế khác nhau do early stopping. Tham số, FLOPs và thời gian không được khớp bằng nhau. RAM là RSS toàn tiến trình lấy mẫu10ms; VRAM là allocator PyTorch, không gồm mọi ứng dụng/context của GPU. Thời gian đo trong phiên desktop hiện tại, không phải máy benchmark được cô lập hoàn toàn; không ngoại suy nguyên số sang máy khác.', '']
    lines += ['## Suy luận', '', '| Mô hình | B1 forward p50 (ms) | B1 end-to-end p50 (ms) | B1 end-to-end p90 (ms) | B64 end-to-end throughput (mẫu/s) | B1 RAM peak (MiB) | B1 VRAM reserved (MiB) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for model in records:
        b = aggregates[model]['benchmarks']
        lines.append(f"| {model} | {fmt(b['1']['forward']['median_ms'])} | {fmt(b['1']['end_to_end']['median_ms'])} | {fmt(b['1']['end_to_end']['p90_ms'])} | {fmt(b['64']['end_to_end']['throughput_samples_per_second'])} | {fmt(b['1']['end_to_end']['ram_peak_mib'])} | {fmt(b['1']['end_to_end']['vram_reserved_mib'])} |")
    lines += ['', 'Mỗi checkpoint/batch chạy trong tiến trình mới,10 warmup+100 lần đo cho từng mode. End-to-end là RSSI trong RAM→tiền xử lý→GPU→dự đoán và inverse→mảng CPU; không gồm quét Wi-Fi, đọc CSV/checkpoint hoặc ghi file. B64 đo throughput; thời gian chia64 không phải latency trực tuyến B1. Không chạy benchmark đồng thời với training.', '',
              '## Chênh lệch với baseline và phân tích lỗi', '', '| Mô hình | Δ mean2D test so với MLP, ghép theo seed (m) |', '|---|---:|']
    baseline = [r['test']['mean_2d'] for r in records['mlp']]
    for model in ['mlp_mixer', 'kan_mixer']:
        difference = [r['test']['mean_2d']-base for r, base in zip(records[model], baseline)]
        lines.append(f'| {model} | {fmt(mean_std(difference))} |')
    lines += ['', 'Δ âm nghĩa là lỗi thấp hơn baseline trong giao thức này. Ba seed là mô tả độ biến thiên do khởi tạo/thứ tự batch; không đủ để tuyên bố ý nghĩa thống kê hay tính ưu việt chung.', '',
              '- `group_metrics.csv`: theo building, floor, building/floor, PHONEID, USERID và số AP phát hiện; mỗi nhóm có số mẫu và mean±SD.',
              '- Test có USERID0 không được ghi nhận; không xem đó là một người. Phân tích theo người thực hiện trên validation, vẫn không chứng minh khái quát sang người mới.',
              '- `worst_test_errors.csv`:20 lỗi lớn nhất mỗi model/seed, kèm nhãn thật/dự đoán, điện thoại và số AP. Dùng để tìm mẫu lỗi; không xóa mẫu khỏi báo cáo.',
              '- `kan_diagnostics_summary.csv`: các lớp KAN ở best và last epoch, lấy từ2 batch train/epoch; không đại diện toàn bộ train/test. `artifacts/*/kan_diagnostics.json` giữ log theo epoch. Ngoài nominal grid không đồng nghĩa mất mọi spline; ngoài full knot support vẫn có nhánh base SiLU.', '',
              '## Giới hạn và bước tiếp theo', '',
              'Split theo nhóm đo là proxy, còn chung người/điểm và một số RSSI giữa train–validation. Official test là một đợt dữ liệu riêng, không đại diện mọi nhà/tòa/thiết bị. Tọa độ dùng quy ước mét của UJIIndoorLoc; chưa xác minh đầy đủ CRS, không suy ra lỗi3D từ nhãn tầng.', '',
              'Nếu cần cải thiện mục tiêu8 m: lập phiên bản thực nghiệm mới chỉ dùng train/validation để chọn token/AP ordering, λxy, dropout, độ rộng, KAN grid và quy tắc split. Vì test này đã được xem, kết quả cải tiến trên cùng test là thăm dò; cần dữ liệu kiểm chứng mới hoặc nested/group CV để có kết luận mạnh hơn. Không gọi các thay đổi đó là kết quả của giao thức đã khóa.', '',
              '## Hình và bằng chứng', '',
              '![Learning curves](figures/learning_curves.png)', '',
              '![CDF lỗi test](figures/test_error_cdf.png)', '',
              '![Lỗi theo building/floor](figures/test_error_by_building_floor.png)', '',
              'Các confusion matrix chuẩn hóa theo hàng, trung bình ba seed, nằm ở `figures/confusion_*.png`; JSON giữ các ma trận count gốc của từng seed. `fairness_audit.json`, `protocol_lock.json`, hash checkpoint và dự đoán từng dòng cho phép truy nguồn bảng.', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'REPORT SAVED: {output / "README.md"}', flush=True)
    return aggregates
