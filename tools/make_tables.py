"""Paper tables and Figure 1 from the run outputs.

Usage: python tools/make_tables.py --pi <results dir> --logs <logs dir> --x86 runs/x86 --out tables
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrnetq.runs import latency, quality  # noqa: E402

NAME = {'fp32': 'FP32', 'fp16': 'FP16', 'int8_minmax_pc': 'INT8 MinMax (per-channel)',
        'int8_minmax_pt': 'INT8 MinMax (per-tensor)', 'int8_percentile_pc': 'INT8 Percentile 99.99',
        'int8_entropy2048_pc': 'INT8 Entropy (KL, 2048 bins)', 'int8_maxslice_pc': 'INT8 MinMax, max-slice calib.',
        'int8_conv1fp32_pc': 'INT8, conv1 in FP32'}
SEEDS = (0, 1, 2)


def rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


class Tables:
    def __init__(self, pi, logs, x86, out):
        self.pi, self.x86, self.out = Path(pi), Path(x86), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        r = rows(Path(logs) / 'pmic.csv')
        self.pt, self.pw = np.array([float(x['t']) for x in r]), np.array([float(x['watts']) for x in r])
        self.idle_windows = [tuple(map(float, line.split(','))) for line in open(Path(logs) / 'idle_windows.csv')]

    def preds(self, run, root=None):
        return {r['case']: r for r in rows((root or self.pi) / run / 'predictions.csv')}

    def quality(self, run, ref, root=None):
        p, r = self.preds(run, root), self.preds(ref, root)
        cs = sorted(p)
        y = np.array([int(p[c]['y']) for c in cs])
        pq = np.array([float(p[c]['prob']) for c in cs])
        pr = np.array([float(r[c]['prob']) for c in cs])
        return {'auc': roc_auc_score(y, pq), 'auc_ref': roc_auc_score(y, pr),
                'flips': sum(p[c]['pred'] != r[c]['pred'] for c in cs), 'mae': float(np.abs(pq - pr).mean()),
                'sens': float(np.mean([int(p[c]['pred']) for c in cs if p[c]['y'] == '1']))}

    def idle_w(self):
        return float(np.median([np.median(self.pw[(self.pt > a + 5) & (self.pt < b)]) for a, b in self.idle_windows]))

    def cost(self, run):
        rec = json.loads((self.pi / run / 'record.json').read_text())
        T = rows(self.pi / run / 'timing.csv')
        js = []
        for r in T:
            a, b = float(r['wall_start']), float(r['wall_end'])
            m = (self.pt >= a) & (self.pt <= b)
            pw = self.pw[m].mean() if m.any() else self.pw[np.argmin(abs(self.pt - (a + b) / 2))]
            js.append(pw * (b - a))
        return {'ms': float(np.median([float(r['t_total']) for r in T]) * 1000), 'rss': rec['peak_rss_mb'],
                'mb': sum(rec['model_bytes'].values()) / 2 ** 20, 'J': float(np.median(js))}

    def write(self, name, header, body, caption, label, colspec):
        with open(self.out / f'{name}.csv', 'w', newline='') as f:
            csv.writer(f).writerows([header] + body)
        tx = lambda s: str(s).replace('±', '$\\pm$').replace('Δ', '$\\Delta$')
        lines = ['\\begin{table}[t]', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
                 '\\begin{adjustbox}{max width=\\columnwidth}', f'\\begin{{tabular}}{{{colspec}}}', '\\toprule',
                 ' & '.join(tx(h) for h in header) + ' \\\\', '\\midrule']
        lines += [' & '.join(tx(x) for x in r) + ' \\\\' for r in body]
        lines += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\end{table}']
        (self.out / f'{name}.tex').write_text('\n'.join(lines) + '\n')

    def seeds(self, backbone, v, root=None):
        return [self.quality(quality(backbone, s, 'test', v), quality(backbone, s, 'test', 'fp32'), root)
                for s in SEEDS if ((root or self.pi) / quality(backbone, s, 'test', v)).exists()]

    def precision(self):
        body = []
        for v in ('fp32', 'fp16', 'int8_minmax_pc'):
            q = self.seeds('alexnet', v)
            c4, c1 = self.cost(latency(v, 0, 4)), self.cost(latency(v, 0, 1))
            aucs = [x['auc'] for x in q]
            body.append([NAME[v], f'{np.mean(aucs):.3f} ± {np.std(aucs):.3f}', '/'.join(str(x['flips']) for x in q),
                         f"{c4['ms'] / 1000:.2f}", f"{c1['ms'] / 1000:.2f}", f"{c4['rss']:.0f}", f"{c4['mb']:.1f}",
                         f"{c4['J']:.1f}"])
        self.write('T_precision', ['Model', 'AUC (3 seeds)', 'Flips', 'Lat. 4T (s)', 'Lat. 1T (s)', 'RAM (MB)',
                                   'Size (MB)', 'Energy (J)'], body, 'Precision on Raspberry Pi 5 (all slices).',
                   'tab:precision', 'lccccccc')

    def calibration(self):
        body = []
        for v in ('int8_minmax_pc', 'int8_minmax_pt', 'int8_percentile_pc', 'int8_entropy2048_pc', 'int8_maxslice_pc',
                  'int8_conv1fp32_pc'):
            q = self.seeds('alexnet', v)
            d = [x['auc'] - x['auc_ref'] for x in q]
            body.append([NAME[v], len(q), f'{np.mean(d):+.3f}', '/'.join(str(x['flips']) for x in q),
                         f"{np.mean([x['mae'] for x in q]):.4f}", f"{np.mean([x['sens'] for x in q]):.2f}"])
        self.write('T_calibration', ['INT8 variant', 'Seeds', 'ΔAUC', 'Flips', 'MAE(p)', 'Sens.'], body,
                   'INT8 calibration choices.', 'tab:calib', 'lccccc')

    def slices(self):
        body, pts = [], []
        for k in (0, 24, 16, 8):
            for v in ('fp32', 'fp16', 'int8_minmax_pc'):
                q = self.quality(quality('alexnet', 0, 'test', v, k), quality('alexnet', 0, 'test', 'fp32'))
                c4, c1 = self.cost(latency(v, k, 4)), self.cost(latency(v, k, 1))
                label = NAME[v].split(' (')[0]
                body.append(['all' if k == 0 else k, label, f"{q['auc']:.3f}", q['flips'], f"{c4['ms']:.0f}",
                             f"{c1['ms']:.0f}", f"{c4['rss']:.0f}", f"{c4['J']:.2f}"])
                pts.append({'k': k, 'prec': label, 'auc': q['auc'], 'ms4': c4['ms'], 'J': c4['J']})
        self.write('T_slices', ['K', 'Precision', 'AUC', 'Flips', 'ms (4T)', 'ms (1T)', 'RAM', 'J'], body,
                   'Slice count and precision (seed 0).', 'tab:slices', 'llcccccc')
        return pts

    def x86_vs_arm(self):
        def row(name, v, root):
            q = self.seeds('alexnet', v, root)
            d = np.mean([x['auc'] - x['auc_ref'] for x in q]) + 0.0
            return [name, f"{np.mean([x['mae'] for x in q]):.4f}", '/'.join(str(x['flips']) for x in q),
                    f'{d:+.3f}'.replace('-0.000', '+0.000')]
        body = [row('Pi 5 (Arm, SDOT)', 'int8_minmax_pc', self.pi),
                row('Xeon (AVX2, no VNNI)', 'int8_minmax_pc', self.x86),
                row('Xeon + reduce\\_range', 'int8_minmax_pc_rr', self.x86)]
        self.write('T_x86_arm', ['CPU', 'MAE(p)', 'Flips', 'ΔAUC'], body, 'Same INT8 model on two CPUs.', 'tab:x86',
                   'lccc')

    def backbone(self):
        body = []
        for bb, name in (('alexnet', 'AlexNet'), ('mobilenet_v3_small', 'MobileNetV3-S')):
            for v in ('fp32', 'int8_minmax_pc'):
                q = self.seeds(bb, v)
                c = self.cost(latency(v, 0, 4) if bb == 'alexnet' else quality(bb, 0, 'test', v))
                body.append([name, 'FP32' if v == 'fp32' else 'INT8', f"{np.mean([x['auc'] for x in q]):.3f}",
                             '/'.join(str(x['flips']) for x in q), f"{c['ms'] / 1000:.2f}", f"{c['rss']:.0f}",
                             f"{c['mb']:.1f}"])
        self.write('T_backbone', ['Backbone', 'Prec.', 'AUC', 'Flips', 'Lat. (s)', 'RAM', 'Size'], body,
                   'Backbone comparison on Pi 5.', 'tab:backbone', 'llccccc')


def figure(pts, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {'FP32': '#2a78d6', 'FP16': '#eb6834', 'INT8 MinMax': '#1baf7a'}
    mk = {'FP32': 'o', 'FP16': 's', 'INT8 MinMax': '^'}
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, key, xl in ((axes[0], 'ms4', 'Latency per exam, 4 threads (ms, log)'), (axes[1], 'J', 'Energy per exam (J, log)')):
        for p in ('FP32', 'FP16', 'INT8 MinMax'):
            s = sorted([x for x in pts if x['prec'] == p], key=lambda x: x[key])
            ax.plot([x[key] for x in s], [x['auc'] for x in s], color=col[p], lw=1.5, marker=mk[p], ms=6,
                    mec='white', mew=1, label=p, zorder=3)
            if p == 'INT8 MinMax':
                for x in s:
                    ax.annotate(f"K={'all' if x['k'] == 0 else x['k']}", (x[key], x['auc']), textcoords='offset points',
                                xytext=(5, -9), fontsize=6.5, color='#444444')
        ax.set_xscale('log')
        ax.set_xlabel(xl)
        ax.grid(axis='y', color='#e5e5e3', lw=0.6)
    lo, hi = min(x['auc'] for x in pts), max(x['auc'] for x in pts)
    axes[0].set_ylim(lo - 0.006, hi + 0.004)
    axes[0].set_ylabel('AUC (ACL, seed 0)')
    axes[0].legend(frameon=False, loc='lower right', fontsize=7)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(Path(out) / f'pareto.{ext}', dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pi', required=True)
    ap.add_argument('--logs', required=True)
    ap.add_argument('--x86', required=True)
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    t = Tables(a.pi, a.logs, a.x86, a.out)
    t.precision()
    t.calibration()
    pts = t.slices()
    t.x86_vs_arm()
    t.backbone()
    figure(pts, a.out)
    json.dump({'idle_W': t.idle_w(), 'points': pts}, open(Path(a.out) / 'summary.json', 'w'), indent=1)
    print('tables written to', a.out)


if __name__ == '__main__':
    main()
