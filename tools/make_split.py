"""Stratified internal train/validation split of the official training set (configs/split_manifest.json)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrnetq.common import DATA, SPLIT, read_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(SPLIT))
    a = ap.parse_args()
    y = read_labels('train')
    rng = np.random.default_rng(a.seed)
    ival = []
    for cls in (0, 1):
        ids = np.array(sorted(i for i, v in y.items() if v == cls))
        rng.shuffle(ids)
        ival += ids[:int(round(len(ids) * a.frac))].tolist()
    ival = sorted(ival)
    itrain = sorted(set(y) - set(ival))
    yt = read_labels('valid')
    m = {'seed': a.seed, 'frac': a.frac, 'task': 'acl',
         'labels_sha256': hashlib.sha256((DATA / 'train-acl.csv').read_bytes()).hexdigest(),
         'counts': {'itrain': [len(itrain), sum(y[i] for i in itrain)], 'ival': [len(ival), sum(y[i] for i in ival)],
                    'test': [len(yt), sum(yt.values())]},
         'itrain': itrain, 'ival': ival}
    Path(a.out).write_text(json.dumps(m))
    print(json.dumps(m['counts']))


if __name__ == '__main__':
    main()
