"""Export one seed: FP32 ONNX of the CNN with a one-channel input, linear heads, plane fusion and threshold.

Usage: python -m mrnetq.export --seed 0 [--backbone alexnet]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .common import PLANES, ROOT, head_forward, load_volume, preprocess, read_labels, split_cases
from .model import GrayFeatures, MRNetPlane


def youden_cv(X, Y, seed):
    """Youden threshold on 5-fold cross-validated scores."""
    k = int(min(5, Y.sum(), len(Y) - Y.sum()))
    cv = cross_val_predict(LogisticRegression(C=1.0), X, Y, cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                           method='decision_function')
    fpr, tpr, thr = roc_curve(Y, cv)
    return float(thr[np.argmax(tpr - fpr)])


def fit_fusion(run_dirs, seed):
    split, ival = split_cases('ival')
    y = read_labels(split)
    logits = {p: json.loads((run_dirs[p] / 'ival_logits.json').read_text()) for p in PLANES}
    X = np.array([[logits[p][str(c)] for p in PLANES] for c in ival])
    Y = np.array([y[c] for c in ival])
    lr = LogisticRegression(C=1.0).fit(X, Y)
    return {'planes': list(PLANES), 'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]),
            'threshold_logit': youden_cv(X, Y, seed), 'threshold_rule': 'Youden J on 5-fold CV scores, internal val',
            'n_ival': len(Y), 'pos_ival': int(Y.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--backbone', default='alexnet')
    ap.add_argument('--runs', default=str(ROOT / 'runs/train'))
    ap.add_argument('--out', default=str(ROOT / 'runs/models'))
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--check', type=int, default=8)
    a = ap.parse_args()
    runs = {p: Path(a.runs) / f'{a.backbone}_{p}_seed{a.seed}' for p in PLANES}
    out = Path(a.out) / f'{a.backbone}_seed{a.seed}'
    (out / 'fp32').mkdir(parents=True, exist_ok=True)
    head, report = {}, {}
    split, ival = split_cases('ival')
    for p in PLANES:
        m = MRNetPlane(a.backbone, pretrained=False)
        m.load_state_dict(torch.load(runs[p] / 'best.pt', map_location='cpu'))
        m.eval()
        head[p] = {'w': m.classifier.weight[0].tolist(), 'b': float(m.classifier.bias[0])}
        g = GrayFeatures(m).eval()
        f = out / 'fp32' / f'{p}.onnx'
        torch.onnx.export(g, torch.zeros(5, 1, 224, 224), str(f), input_names=['x'], output_names=['feats'],
                          dynamic_axes={'x': {0: 'S'}, 'feats': {0: 'S'}}, opset_version=a.opset, dynamo=False)
        sess = ort.InferenceSession(str(f), providers=['CPUExecutionProvider'])
        diffs = []
        for c in ival[:a.check]:
            x = preprocess(load_volume(split, p, c))
            with torch.no_grad():
                lt = m(torch.from_numpy(x)).item()
            lo = head_forward(sess.run(None, {'x': x})[0], head[p]['w'], head[p]['b'])
            diffs.append(abs(lt - lo))
        report[p] = {'max_abs_logit_diff': float(max(diffs)), 'n': len(diffs)}
        assert max(diffs) < 1e-3, f'{p}: ONNX differs from PyTorch by {max(diffs)}'
    (out / 'head.json').write_text(json.dumps(head))
    fusion = fit_fusion(runs, a.seed)
    (out / 'fusion.json').write_text(json.dumps(fusion, indent=1))
    (out / 'export_record.json').write_text(json.dumps({'args': vars(a), 'parity': report,
                                                        'torch': torch.__version__, 'ort': ort.__version__}, indent=1))
    print(json.dumps({'parity': report, 'fusion': {k: fusion[k] for k in ('coef', 'intercept', 'threshold_logit')}}))


if __name__ == '__main__':
    main()
