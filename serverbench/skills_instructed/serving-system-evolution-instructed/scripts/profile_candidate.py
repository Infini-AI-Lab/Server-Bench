#!/usr/bin/env python3
"""Capture a bounded GPU profile of an immutable candidate, outside scored runs.

The driver defines setup(candidate_dir) -> state and run(state). setup loads and
warms the actual candidate; run exercises representative prefill/decode shapes.
For multi-process engines choose nsys/ncu and make run wait for all GPU work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import sys

from run_trial import digest_tree, execute, write_json


def sha(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gpu_events(backend, path):
    """Inspect native exports; a successful profiler exit alone proves nothing."""
    if backend == 'torch':
        body = json.loads(Path(path).read_text())
        events = [e for e in body.get('traceEvents', [])
                  if e.get('cat') == 'kernel' and e.get('dur', 0) > 0]
        return len(events)
    if backend == 'nsys':
        with sqlite3.connect(f'file:{Path(path)}?mode=ro', uri=True) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            return sum(db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                       for name in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL')
                       if name in tables)
    rows = list(csv.reader(io.StringIO(Path(path).read_text())))
    header = next((i for i, row in enumerate(rows) if 'Kernel Name' in row and 'ID' in row), None)
    if header is None:
        return 0
    index = rows[header].index('ID')
    return len({row[index] for row in rows[header + 1:]
                if len(row) > index and row[index].isdigit()})


def worker(args):
    import torch
    candidate = args.candidate_dir.resolve()
    sys.path.insert(0, str(candidate))
    spec = importlib.util.spec_from_file_location('profile_driver', args.driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = module.setup(candidate)
    for _ in range(args.warmup):
        module.run(state)
    torch.cuda.synchronize()
    if args.backend == 'torch':
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as capture:
            for _ in range(args.repeats):
                module.run(state)
            torch.cuda.synchronize()
        capture.export_chrome_trace(str(args.run_dir / 'trace.json'))
    else:
        # nsys/ncu wrapper captures children as well. setup/warmup is outside
        # this explicit region, and the same frozen driver is used for A/B.
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.repeats):
            module.run(state)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    loaded = sorted({str(Path(m.__file__).resolve().relative_to(candidate))
                     for m in list(sys.modules.values())
                     if getattr(m, '__file__', None)
                     and Path(m.__file__).resolve().is_relative_to(candidate)})
    write_json(args.run_dir / 'worker.json', {'candidate_modules': loaded,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=('torch', 'nsys', 'ncu'), default='torch')
    parser.add_argument('--workload', required=True, help='phase, ISL/OSL, batch, dtype, device topology')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args)
        return 0
    if args.warmup < 1 or args.repeats < 1 or not 0 < args.timeout <= 900:
        parser.error('warmup/repeats positive; timeout at most 900 seconds')
    source, run = args.candidate_dir.resolve(), args.run_dir.resolve()
    if run.is_relative_to(source) or not (source / 'serve.sh').is_file():
        parser.error('profile outside a deployable candidate directory')
    run.mkdir(parents=True, exist_ok=False)
    files = digest_tree(source)
    # Copy the driver to bind the experiment, but never add it to deployed code.
    driver = run / 'driver.py'
    driver.write_bytes(args.driver.read_bytes())
    cmd = [sys.executable, str(Path(__file__).resolve()), '--worker',
           '--candidate-dir', str(source), '--driver', str(driver), '--run-dir', str(run),
           '--backend', args.backend, '--workload', args.workload,
           '--warmup', str(args.warmup), '--repeats', str(args.repeats)]
    native = run / 'trace.json'
    if args.backend == 'nsys':
        cmd = ['nsys', 'profile', '--sample=none', '--trace=cuda,nvtx',
               '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
               '--output', str(run / 'capture'), *cmd]
        native = run / 'capture.nsys-rep'
    elif args.backend == 'ncu':
        cmd = ['ncu', '--target-processes', 'all', '--profile-from-start', 'off',
               '--launch-count', '8', '--set', 'basic', '--export', str(run / 'capture'), *cmd]
        native = run / 'capture.ncu-rep'
    report = {'schema': 'serving_evolution.profile/1', 'backend': args.backend,
              'candidate_tree_sha': sha(files), 'files': files,
              'driver_sha256': file_sha(driver), 'workload': args.workload,
              'warmup': args.warmup, 'repeats': args.repeats, 'status': 'failed',
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'scope': 'instrumented diagnosis only; not scored performance'}
    try:
        report.update(execute(cmd, cwd=source, env={**os.environ, 'APP_DIR': str(source),
                              'PYTHONDONTWRITEBYTECODE': '1'}, log=run / 'profile.log',
                              timeout=args.timeout))
        parsed = native
        if args.backend == 'nsys':
            parsed = run / 'capture.sqlite'
            conversion = execute(['nsys', 'export', '--type', 'sqlite', '--output', str(parsed),
                                  str(native)], cwd=run, env=dict(os.environ),
                                 log=run / 'export.log', timeout=60)
            if conversion['exit_code']:
                raise ValueError('nsys export failed')
        elif args.backend == 'ncu':
            parsed = run / 'kernels.csv'
            conversion = execute(['ncu', '--import', str(native), '--csv', '--page', 'raw'],
                                 cwd=run, env=dict(os.environ), log=parsed, timeout=60)
            if conversion['exit_code']:
                raise ValueError('ncu export failed')
        count = gpu_events(args.backend, parsed)
        worker_data = json.loads((run / 'worker.json').read_text())
        report.update(worker_data, gpu_kernel_events=count,
                      artifacts={str(p): file_sha(p) for p in {native, parsed, driver}})
        unchanged = digest_tree(source) == files
        if (report['exit_code'] == 0 and not report['timed_out'] and count > 0
                and worker_data['candidate_modules'] and unchanged):
            report['status'] = 'passed'
        else:
            report['why'] = 'require live candidate modules, CUDA kernels, successful exit and unchanged source'
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        report['why'] = str(exc)
    write_json(run / 'profile.json', report)
    print(json.dumps({'status': report['status'], 'report': str(run / 'profile.json')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
