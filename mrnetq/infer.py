"""Exam-level inference with ONNX Runtime on CPU, with per-stage timing.

Usage: python -m mrnetq.infer --bundle runs/models/alexnet_seed0 --variant int8_minmax_pc --split test --out runs/demo
"""
import argparse
import csv
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .common import PLANES, canonical, env_info, load_volume, preprocess, read_labels, split_cases


def vcgencmd(*args):
    try:
        return subprocess.run(['vcgencmd', *args], capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def read_sys(path, scale=1.0):
    try:
        return int(open(path).read()) / scale
    except (OSError, ValueError):
        return None


def temp_c():
    return read_sys('/sys/class/thermal/thermal_zone0/temp', 1000)


def freq_mhz():
    return read_sys('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 1000)


def cool_down(above, to, max_wait=900):
    """Sleep until the SoC is at `to` if it is above `above`; returns the seconds waited."""
    t = temp_c()
    if t is None or t <= above:
        return 0.0
    t0 = time.time()
    while (temp_c() or 0) > to and time.time() - t0 < max_wait:
        time.sleep(2)
    return time.time() - t0


def make_session(path, threads, spin, profile_dir=None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry('session.intra_op.allow_spinning', '1' if spin else '0')
    if profile_dir:
        so.enable_profiling = True
        so.profile_file_prefix = str(Path(profile_dir) / Path(path).stem)
    return ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])


def kernel_summary(profile_json):
    """Executed kernels by op, provider and input types."""
    ev = json.loads(Path(profile_json).read_text())
    cnt = {}
    for e in ev:
        if e.get('cat') == 'Node' and e['name'].endswith('_kernel_time'):
            a = e.get('args', {})
            types = ','.join(sorted({t for d in a.get('input_type_shape', []) for t in d}))
            key = f"{a.get('op_name')}|{a.get('provider')}|{types}"
            cnt[key] = cnt.get(key, 0) + 1
    return cnt


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--variant', default='fp32')
    ap.add_argument('--split', default='test', choices=['test', 'ival'])
    ap.add_argument('--k', type=int, default=0, help='slices per plane (0 = all)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--spin', type=int, default=0, help='ORT spin-wait; off by default for fair energy measurement')
    ap.add_argument('--save-feats', action='store_true', help='store per-slice features (H4 analysis)')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--cool-above', type=float, default=68.0, help='deg C; wait before an exam when hotter')
    ap.add_argument('--cool-to', type=float, default=60.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    a.variant = canonical(a.variant)

    out = Path(a.out)
    if (out / 'record.json').exists() and json.loads((out / 'record.json').read_text()).get('status') == 'done':
        print('already done:', out)
        return
    out.mkdir(parents=True, exist_ok=True)
    b = Path(a.bundle)
    head = json.loads((b / 'head.json').read_text())
    fus = json.loads((b / 'fusion.json').read_text())
    split, cases = split_cases(a.split)
    if a.limit:
        cases = cases[:a.limit]
    y = read_labels(split)
    rec = {'args': vars(a), 'env': env_info(), 'n_cases': len(cases), 'status': 'running',
           'temp_start': vcgencmd('measure_temp'), 'throttled_start': vcgencmd('get_throttled'),
           'model_bytes': {p: (b / a.variant / f'{p}.onnx').stat().st_size for p in PLANES}}
    t0 = time.perf_counter()
    sess = {p: make_session(b / a.variant / f'{p}.onnx', a.threads, a.spin, out if a.profile else None) for p in PLANES}
    rec['session_init_sec'] = round(time.perf_counter() - t0, 3)
    W = {p: (np.asarray(head[p]['w'], np.float32), head[p]['b']) for p in PLANES}
    coef, icpt, thr = np.asarray(fus['coef']), fus['intercept'], fus['threshold_logit']

    def run_exam(c, keep=False):
        tl = time.perf_counter()
        vols = {p: load_volume(split, p, c) for p in PLANES}
        tp = time.perf_counter()
        xs = {p: preprocess(vols[p], a.k) for p in PLANES}
        tc = time.perf_counter()
        feats = {p: sess[p].run(None, {'x': xs[p]})[0] for p in PLANES}
        th = time.perf_counter()
        logit = np.array([float(feats[p].max(0) @ W[p][0] + W[p][1]) for p in PLANES])
        fused = float(logit @ coef + icpt)
        te = time.perf_counter()
        t = {'t_load': tp - tl, 't_pre': tc - tp, 't_cnn': th - tc, 't_head': te - th, 't_total': te - tl,
             'n_slices': sum(len(xs[p]) for p in PLANES)}
        return logit, fused, t, (feats if keep else None)

    for c in cases[:a.warmup]:
        run_exam(c)
    preds, timing, featstore = [], [], {}
    cooled = 0.0
    for r in range(a.repeats):
        for c in cases:
            w = cool_down(a.cool_above, a.cool_to)
            cooled += w
            if w > 0:
                run_exam(c)  # unmeasured exam to bring the clock back up
            t_before = temp_c()
            ws = time.time()
            logit, fused, t, feats = run_exam(c, keep=(r == 0 and a.save_feats))
            timing.append({'case': c, 'rep': r, **{k: round(v, 6) for k, v in t.items()},
                           'wall_start': round(ws, 4), 'wall_end': round(time.time(), 4),
                           'temp_before': t_before, 'temp_after': temp_c(), 'freq_mhz_end': freq_mhz()})
            if r == 0:
                preds.append({'case': c, 'y': y[c], **{f'logit_{p}': float(v) for p, v in zip(PLANES, logit)},
                              'fused_logit': fused, 'prob': float(1 / (1 + np.exp(-fused))), 'pred': int(fused >= thr)})
                if feats is not None:
                    for p in PLANES:
                        featstore[f'{p}_{c}'] = feats[p].astype(np.float32)
    write_csv(out / 'predictions.csv', preds)
    write_csv(out / 'timing.csv', timing)
    if featstore:
        np.savez_compressed(out / 'feats.npz', **featstore)
    if a.profile:
        rec['kernels'] = {p: kernel_summary(sess[p].end_profiling()) for p in PLANES}
    tot = np.array([t['t_total'] for t in timing])
    fq = [t['freq_mhz_end'] for t in timing if t['freq_mhz_end']]
    tp = [t['temp_after'] for t in timing if t['temp_after']]
    rec.update(cooldown_sec=round(cooled, 1), freq_mhz_min=min(fq) if fq else None, temp_max=max(tp) if tp else None)
    rec.update(status='done', peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               t_total_median=float(np.median(tot)), t_total_p95=float(np.percentile(tot, 95)),
               temp_end=vcgencmd('measure_temp'), throttled_end=vcgencmd('get_throttled'))
    (out / 'record.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ('peak_rss_mb', 't_total_median', 't_total_p95', 'session_init_sec')}))


if __name__ == '__main__':
    main()
