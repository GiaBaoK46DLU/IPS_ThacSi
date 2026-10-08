"""Latency and memory on fixed training fingerprints; no quality tuning on test."""
from pathlib import Path
import time

import numpy as np
import torch

from .data import feature_transform, sha256
from .metrics import inverse_xy
from .predict import read_fingerprints
from .resources import MemorySampler, cuda_memory, synchronize
from .runtime import configure_runtime, load_checkpoint, save_json, software_info


def latency_statistics(seconds, batch_size):
    values = np.asarray(seconds, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Latency needs at least two positive finite measurements.')
    mean = float(values.mean())
    return {'repeats': len(values), 'mean_ms': 1000 * mean,
            'std_ms': 1000 * float(values.std(ddof=1)),
            'median_ms': 1000 * float(np.median(values)), 'p90_ms': 1000 * float(np.quantile(values, .9)),
            'throughput_samples_per_second': batch_size / mean,
            'amortized_mean_ms_per_sample': 1000 * mean / batch_size,
            'raw_seconds': values.tolist()}


@torch.inference_mode()
def benchmark(checkpoint_path: Path, input_path: Path, output_path: Path, batch_size=1,
              warmup=10, repeats=100, device_name='cuda'):
    if output_path.exists():
        raise FileExistsError(f'Benchmark already exists: {output_path}')
    if batch_size < 1 or warmup < 1 or repeats < 2:
        raise ValueError('Invalid benchmark batch/warmup/repeats.')
    device = configure_runtime(0, device_name, threads=4)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    # Optimizer state belongs to training, not a deployed inference model.
    checkpoint.pop('optimizer_state_dict', None)
    rssi = read_fingerprints(input_path, checkpoint['ap_order'])
    if len(rssi) < batch_size:
        raise ValueError('Not enough fixed benchmark fingerprints.')
    rssi = rssi[:batch_size].copy()
    prepared = torch.from_numpy(feature_transform(rssi)).to(device)
    pairs = np.asarray([[c['building'], c['floor']] for c in checkpoint['classes']], dtype=np.int64)

    def forward():
        return model(prepared)

    def end_to_end():
        features = torch.from_numpy(feature_transform(rssi)).to(device)
        output = model(features)
        confidence, predicted = output['logits'].softmax(dim=1).max(dim=1)
        ids = predicted.cpu().numpy()
        coordinates = inverse_xy(output['xy'].cpu().numpy(), checkpoint['preprocessing'])
        return pairs[ids], coordinates, confidence.cpu().numpy()

    result = {'model': checkpoint['config']['model'], 'seed': checkpoint['seed'],
              'checkpoint_sha256': sha256(checkpoint_path), 'input_sha256': sha256(input_path),
              'batch_size': batch_size, 'warmup': warmup, 'repeats': repeats,
              'software': software_info(device), 'precision': 'FP32', 'modes': {},
              'forward_scope': 'precomputed input on device -> logits and normalized xy; synchronized wall time',
              'end_to_end_scope': 'in-memory RSSI -> features -> device transfer -> model -> softmax/class mapping/inverse xy -> CPU arrays',
              'excluded': 'Wi-Fi scanning, CSV/checkpoint loading, process startup, network and output file writing',
              'memory_scope': 'resident inference process, including Python/native libraries and fixed inputs; optimizer state discarded; warmup included'}
    for mode, call in [('forward', forward), ('end_to_end', end_to_end)]:
        synchronize(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        memory = MemorySampler().start()
        try:
            for _ in range(warmup):
                value = call()
                del value
            synchronize(device)
            durations = []
            for _ in range(repeats):
                synchronize(device)
                started = time.perf_counter()
                value = call()
                synchronize(device)
                durations.append(time.perf_counter() - started)
                del value
            result['modes'][mode] = {**latency_statistics(durations, batch_size),
                                     'ram': memory.stop(), 'cuda': cuda_memory(device)}
        finally:
            if not memory.stop_event.is_set():
                memory.stop()
    check = forward()
    if any(not torch.isfinite(value).all() for value in check.values()):
        raise FloatingPointError('Non-finite benchmark output.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, result)
    print(f"BENCHMARK {result['model']} seed={result['seed']} batch={batch_size}: "
          f"forward p50={result['modes']['forward']['median_ms']:.3f}ms, "
          f"end_to_end p50={result['modes']['end_to_end']['median_ms']:.3f}ms", flush=True)
    return result
