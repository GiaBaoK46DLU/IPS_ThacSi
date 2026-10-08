"""Train the frozen nine-run suite, streaming each epoch to a visible terminal."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from ips.data import ROOT, sha256
from ips.runtime import code_hashes, read_config


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs/stage7_cuda_v1')
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f'Suite already exists: {root}; use a new directory.')
    root.mkdir(parents=True)
    config_path = ROOT / 'configs/base.json'
    config = read_config(config_path)
    hashes = code_hashes()
    training_files = ['data.py', 'dataset.py', 'train.py', 'metrics.py', 'runtime.py', 'resources.py',
                      'diagnostics.py', 'models/networks.py', 'models/vendor/efficient_kan.py']
    protocol = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_head_at_start': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        'models': ['mlp', 'mlp_mixer', 'kan_mixer'], 'seeds': [42, 43, 44], 'device': 'cuda',
        'config': config, 'config_sha256': sha256(config_path),
        'training_implementation_sha256': {'src/ips/' + name: hashes['src/ips/' + name] for name in training_files},
        'split_manifest_sha256': sha256(ROOT / config['split_dir'] / 'manifest.json'),
        'selection': 'internal_validation_mean_2d_with_min_delta_0.01',
        'target': {'metric': 'mean_2d_all_1111_official_test_rows', 'maximum_metres': 8},
        'benchmark': {'batch_sizes': [1, 64], 'warmup': 10, 'repeats': 100,
                      'scope': 'forward_and_in_memory_end_to_end_separately'},
        'test_policy': 'Evaluate nine fixed best-validation checkpoints only after all training passes fairness audit; no retuning on test.',
    }
    write_json(root / 'protocol_lock.json', protocol)
    frozen_config = root / 'frozen_config.json'
    write_json(frozen_config, config)
    state = {'status': 'training', 'pid': os.getpid(), 'completed': [], 'current_run': None,
             'test_evaluated': False}
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    with (root / 'terminal.log').open('w', encoding='utf-8', buffering=1) as log:
        def emit(line):
            print(line, flush=True)
            log.write(line + '\n')

        emit('STAGE 7 | RTX 4060 Ti | 3 models x 3 seeds | max 100 epochs | validation only')
        emit(f'Logs: {root}')
        try:
            for model in protocol['models']:
                for seed in protocol['seeds']:
                    name = f'{model}_seed{seed}'
                    state['current_run'] = name
                    write_json(root / 'suite_status.json', state)
                    emit(f'\n===== RUN {len(state["completed"]) + 1}/9: {name} =====')
                    command = [sys.executable, '-u', '-m', 'ips', 'train', '--config', str(frozen_config),
                               '--model', model, '--seed', str(seed), '--device', 'cuda',
                               '--epochs', '100', '--run-dir', str(root / name)]
                    with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
                        state['child_pid'] = process.pid
                        write_json(root / 'suite_status.json', state)
                        for line in process.stdout:
                            emit(line.rstrip('\r\n'))
                        if process.wait() != 0:
                            raise RuntimeError(f'{name} failed; see terminal.log.')
                    state['completed'].append(name)
                    write_json(root / 'suite_status.json', state)
                    summary = json.loads((root / name / 'summary.json').read_text(encoding='utf-8'))
                    emit(f'FINISHED {name}: epochs={summary["epochs_completed"]}, best_epoch={summary["best_epoch"]}, '
                         f'validation_mean2d={summary["best_validation_mean_2d"]:.3f}m, '
                         f'training_seconds={summary["training_seconds"]:.1f}')
            state.update(status='training_completed', current_run=None, child_pid=None)
            emit('\nALL 9 TRAINING RUNS COMPLETED. Next: fairness audit, fixed-checkpoint evaluation and inference benchmarks.')
        except BaseException as error:
            state.update(status='failed_or_interrupted', error=f'{type(error).__name__}: {error}')
            emit(state['error'])
            raise
        finally:
            write_json(root / 'suite_status.json', state)


if __name__ == '__main__':
    main()
