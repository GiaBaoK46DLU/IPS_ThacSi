"""Audit first, then evaluate fixed checkpoints and benchmark sequentially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ips.analysis import audit_suite, report
from ips.data import AP_COLUMNS, ROOT, csv_bytes, sha256
from ips.dataset import load_bundle
from ips.runtime import code_hashes, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-root', type=Path, default=ROOT / 'runs/stage7_cuda_v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/stage7_cuda_v1')
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    if args.output_dir.exists():
        raise FileExistsError('Report already exists; choose a new output directory.')
    audit = audit_suite(root)
    pretest = root / 'pretest_audit.json'
    if pretest.exists():
        previous = json.loads(pretest.read_text(encoding='utf-8'))
        if previous != audit:
            raise ValueError('Audit changed after test evaluation was authorized by the frozen protocol.')
    else:
        save_json(pretest, audit)
    provenance = root / 'evaluation_provenance.json'
    if not provenance.exists():
        save_json(provenance, {'code_sha256': code_hashes(),
            'controller_sha256': sha256(Path(__file__)),
            'note': 'Evaluation implementation recorded separately from the frozen training core.'})
    bundle = load_bundle(ROOT / audit['protocol']['config']['split_dir'])
    indices = bundle.indices['train'][:64]
    benchmark_input = root / 'benchmark_inputs.csv'
    content = csv_bytes(AP_COLUMNS, bundle.development.rssi[indices].tolist())
    if benchmark_input.exists() and benchmark_input.read_bytes() != content:
        raise ValueError('Fixed benchmark input changed.')
    benchmark_input.write_bytes(content)
    save_json(root / 'benchmark_input_provenance.json', {
        'selection': 'first 64 saved internal-train row indices, in original CSV order',
        'row_indices': indices.tolist(), 'source': bundle.manifest['sources']['development'],
        'input_sha256': sha256(benchmark_input), 'no_test_inputs_used': True})
    del bundle
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    state_path = root / 'evaluation_status.json'
    state = {'status': 'evaluating', 'pid': os.getpid(), 'completed': []}
    save_json(state_path, state)
    try:
        with (root / 'evaluation_terminal.log').open('a', encoding='utf-8', buffering=1) as log:
            def run(command):
                print(' '.join(command), flush=True)
                with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                    if process.wait() != 0:
                        raise RuntimeError('Evaluation or benchmark failed; read evaluation_terminal.log.')

            for selected in audit['runs']:
                directory = root / selected['run']
                checkpoint = directory / 'best.pt'
                for split in ['validation', 'test']:
                    destination = directory / split
                    if destination.exists():
                        metrics = json.loads((destination / 'metrics.json').read_text(encoding='utf-8'))
                        if (metrics['checkpoint_sha256'] != selected['checkpoint_sha256'] or
                                metrics['split'] != split or not (destination / 'predictions.csv').exists()):
                            raise ValueError('Cannot resume an evaluation with different checkpoint or incomplete output.')
                    else:
                        run([sys.executable, '-u', '-m', 'ips', 'evaluate', '--checkpoint', str(checkpoint),
                             '--split', split, '--split-dir', str(ROOT / audit['protocol']['config']['split_dir']),
                             '--output-dir', str(destination), '--device', 'cuda'])
                    state['completed'].append(selected['run'] + '/' + split)
                    save_json(state_path, state)
                settings = audit['protocol']['benchmark']
                for batch in settings['batch_sizes']:
                    destination = directory / 'benchmarks' / f'batch{batch}.json'
                    if destination.exists():
                        saved = json.loads(destination.read_text(encoding='utf-8'))
                        if (saved['checkpoint_sha256'] != selected['checkpoint_sha256'] or
                                saved['input_sha256'] != sha256(benchmark_input) or saved['batch_size'] != batch or
                                saved['warmup'] != settings['warmup'] or saved['repeats'] != settings['repeats']):
                            raise ValueError('Cannot resume a benchmark with different provenance or protocol.')
                    else:
                        run([sys.executable, '-u', '-m', 'ips', 'benchmark', '--checkpoint', str(checkpoint),
                             '--csv', str(benchmark_input), '--batch-size', str(batch), '--warmup', str(settings['warmup']),
                             '--repeats', str(settings['repeats']), '--output', str(destination), '--device', 'cuda'])
                    state['completed'].append(selected['run'] + f'/benchmark{batch}')
                    save_json(state_path, state)
        state['status'] = 'reporting'
        save_json(state_path, state)
        report(root, args.output_dir)
        state.update(status='completed', report=str(args.output_dir))
    except BaseException as error:
        state.update(status='failed_or_interrupted', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save_json(state_path, state)


if __name__ == '__main__':
    main()
