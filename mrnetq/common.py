"""Data loading and preprocessing as in the original MRNet loader. NumPy only, so it also runs on the Pi."""
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

PLANES = ('sagittal', 'coronal', 'axial')
INPUT_DIM = 224
MAX_PIXEL = 255.0
MEAN, STD = 58.09, 49.73
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('MRNETQ_DATA', ROOT / 'data/MRNet-v1.0'))
SPLIT = Path(os.environ.get('MRNETQ_SPLIT', ROOT / 'configs/split_manifest.json'))

VARIANTS = ('fp32', 'fp16', 'int8_minmax_pc', 'int8_minmax_pt', 'int8_percentile_pc', 'int8_entropy2048_pc',
            'int8_maxslice_pc', 'int8_conv1fp32_pc', 'int8_minmax_pc_rr')
ALIASES = {'int8_aggaware_pc': 'int8_maxslice_pc', 'int8_mix_c1_pc': 'int8_conv1fp32_pc'}


def canonical(variant):
    return ALIASES.get(variant, variant)


def read_labels(split, task='acl', data=DATA):
    out = {}
    for line in (Path(data) / f'{split}-{task}.csv').read_text().split():
        i, y = line.split(',')
        out[int(i)] = int(y)
    return out


def load_volume(split, plane, case, data=DATA):
    return np.load(Path(data) / split / plane / f'{case:04d}.npy')


def uniform_indices(n, k):
    """k evenly spaced slice indices; all slices when k is 0/None or k >= n."""
    if not k or k >= n:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))


def preprocess(vol, k=None):
    """uint8 (S, 256, 256) -> normalized float32 (S', 1, 224, 224)."""
    vol = vol[uniform_indices(len(vol), k)]
    pad = (vol.shape[2] - INPUT_DIM) // 2
    v = vol[:, pad:pad + INPUT_DIM, pad:pad + INPUT_DIM].astype(np.float32)
    lo, hi = v.min(), v.max()
    v = (v - lo) / max(hi - lo, 1e-6) * MAX_PIXEL
    v = (v - MEAN) / STD
    return v[:, None]


def split_cases(name, manifest=None):
    """itrain/ival from the split manifest; test is the official validation split."""
    if name == 'test':
        return 'valid', sorted(read_labels('valid'))
    m = json.loads(Path(manifest or SPLIT).read_text())
    return 'train', m[name]


def head_forward(feats, w, b):
    return float(feats.max(0) @ np.asarray(w, np.float32) + b)


def env_info():
    info = {'host': platform.node(), 'machine': platform.machine(), 'python': platform.python_version(),
            'numpy': np.__version__, 'time': time.strftime('%F %T')}
    try:
        import onnxruntime as ort
        info['onnxruntime'] = ort.__version__
    except ImportError:
        pass
    return info
