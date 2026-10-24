"""Numbers quoted in the text: H4 feature shift, per-variant thresholds, ECE, PMIC logger overhead.

Usage: python tools/analyze.py {h4,threshold,ece,overhead} --pi <results dir>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrnetq.common import PLANES, uniform_indices  # noqa: E402
from mrnetq.export import youden_cv  # noqa: E402
from mrnetq.runs import overhead, quality  # noqa: E402

VARIANTS = ('fp16', 'int8_minmax_pc', 'int8_minmax_pt', 'int8_percentile_pc', 'int8_entropy2048_pc',
            'int8_maxslice_pc', 'int8_conv1fp32_pc')


def preds(d):
    with open(Path(d) / 'predictions.csv') as f:
        return {r['case']: r for r in csv.DictReader(f)}


def h4(a):
    head = json.loads((Path(a.models) / 'alexnet_seed0/head.json').read_text())
    ref = np.load(Path(a.pi) / quality('alexnet', 0, 'test', 'fp32') / 'feats.npz')
    for v in ('int8_minmax_pc', 'int8_percentile_pc', 'int8_entropy2048_pc'):
        q = np.load(Path(a.pi) / quality('alexnet', 0, 'test', v) / 'feats.npz')
        for k in (0, 24, 16, 8):
            sb, pb, sa, ea = [], [], [], []
            for key in ref.files:
                f, fq = ref[key], q[key]
                idx = uniform_indices(len(f), k)
                f, fq = f[idx], fq[idx]
                w = np.asarray(head[key.split('_')[0]]['w'], np.float32)
                sb.append((fq - f).mean())
                pb.append((fq.max(0) - f.max(0)).mean())
                sa.append(np.abs((fq - f) @ w).mean())
                ea.append(abs((fq.max(0) - f.max(0)) @ w))
            print(f'{v:<22} K={k:<3} shift per slice {np.mean(sb):+.4f}  after max {np.mean(pb):+.4f}  '
                  f'|logit change| exam/slice {np.mean(ea) / max(np.mean(sa), 1e-12):.2f}')


def threshold(a):
    for s in (0, 1, 2):
        fus = json.loads((Path(a.models) / f'alexnet_seed{s}/fusion.json').read_text())
        ref = preds(Path(a.pi) / quality('alexnet', s, 'test', 'fp32'))
        dref = {c: float(r['fused_logit']) >= fus['threshold_logit'] for c, r in ref.items()}
        for v in VARIANTS:
            dv, dt = Path(a.pi) / quality('alexnet', s, 'ival', v), Path(a.pi) / quality('alexnet', s, 'test', v)
            if not (dv.exists() and dt.exists()):
                continue
            iv = preds(dv)
            cs = sorted(iv, key=int)
            X = np.array([[float(iv[c][f'logit_{p}']) for p in PLANES] for c in cs])
            Y = np.array([int(iv[c]['y']) for c in cs])
            tb = youden_cv(X, Y, s)
            p = preds(dt)
            fa = sum((float(p[c]['fused_logit']) >= fus['threshold_logit']) != dref[c] for c in p)
            fb = sum((float(p[c]['fused_logit']) >= tb) != dref[c] for c in p)
            print(f'seed {s} {v:<22} changed decisions: shared threshold {fa:3d}, re-selected threshold {fb:3d}')


def ece(a):
    for v in ('fp32', 'fp16', 'int8_minmax_pc', 'int8_percentile_pc', 'int8_entropy2048_pc'):
        vals = []
        for s in (0, 1, 2):
            p = preds(Path(a.pi) / quality('alexnet', s, 'test', v))
            y = np.array([int(r['y']) for r in p.values()])
            pr = np.array([float(r['prob']) for r in p.values()])
            edges, e = np.linspace(0, 1, 11), 0.0
            for i in range(10):
                m = (pr >= edges[i]) & ((pr < edges[i + 1]) if i < 9 else (pr <= 1))
                if m.any():
                    e += m.mean() * abs(pr[m].mean() - y[m].mean())
            vals.append(e)
        print(f'{v:<22} ECE {np.mean(vals):.3f} ({"/".join(f"{x:.3f}" for x in vals)})')


def overhead_cmd(a):
    for v in ('fp32', 'int8_minmax_pc'):
        med = []
        for on in (True, False):
            with open(Path(a.pi) / overhead(v, on) / 'timing.csv') as f:
                med.append(np.median([float(r['t_total']) for r in csv.DictReader(f)]) * 1000)
        print(f'{v:<16} median {med[0]:.1f} ms with logger, {med[1]:.1f} ms without ({100 * (med[0] / med[1] - 1):+.2f}%)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['h4', 'threshold', 'ece', 'overhead'])
    ap.add_argument('--pi', required=True)
    ap.add_argument('--models', default=str(Path(__file__).resolve().parents[1] / 'runs/models'))
    a = ap.parse_args()
    {'h4': h4, 'threshold': threshold, 'ece': ece, 'overhead': overhead_cmd}[a.what](a)


if __name__ == '__main__':
    main()
