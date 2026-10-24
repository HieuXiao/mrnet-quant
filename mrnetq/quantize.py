"""FP16 and post-training INT8 variants of an exported bundle.

Usage: python -m mrnetq.quantize --bundle runs/models/alexnet_seed0 [--variants ...]
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from .common import PLANES, VARIANTS, canonical, load_volume, preprocess, split_cases

# histogram calibrators need batches of equal size
CHUNK = 16


class Reader(CalibrationDataReader):
    def __init__(self, batches):
        self.it = iter([{'x': b} for b in batches])

    def get_next(self):
        return next(self.it, None)


def calib_batches(plane, cases, fp32_path=None, topk=0):
    """With topk > 0, keep the topk slices of each exam that hold the channel-wise maximum most often."""
    sess = ort.InferenceSession(str(fp32_path), providers=['CPUExecutionProvider']) if topk else None
    out = []
    for c in cases:
        x = preprocess(load_volume('train', plane, c))
        if topk:
            f = sess.run(None, {'x': x})[0]
            wins = np.bincount(f.argmax(0), minlength=len(x))
            x = x[np.sort(np.argsort(-wins, kind='stable')[:topk])]
        out.append(x)
    allx = np.concatenate(out)
    n = len(allx) // CHUNK * CHUNK
    return [allx[i:i + CHUNK] for i in range(0, n, CHUNK)]


def entropy_ranges(pre, batches, num_bins=2048, num_quantized_bins=128):
    # quantize_static has no num_bins option (it uses 128 bins), so the ranges are computed here
    from onnxruntime.quantization.calibrate import create_calibrator
    with tempfile.TemporaryDirectory() as td:
        cal = create_calibrator(str(pre), None, augmented_model_path=str(Path(td) / 'aug.onnx'),
                                calibrate_method=CalibrationMethod.Entropy,
                                extra_options={'symmetric': False, 'num_bins': num_bins,
                                               'num_quantized_bins': num_quantized_bins})
        cal.collect_data(Reader(batches))
        rng = cal.compute_data()
    return {k: [{'rmin': np.float32(rng[k].lowest), 'rmax': np.float32(rng[k].highest)}] for k in rng}


def first_conv_name(model_path):
    m = onnx.load(str(model_path))
    return next(n.name for n in m.graph.node if n.op_type == 'Conv')


def build(v, src, pre, dst, plane, cases, a):
    if v == 'fp16':
        from onnxconverter_common import float16
        onnx.save(float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True), str(dst))
        return
    rr = v.endswith('_rr')
    base = v[:-3] if rr else v
    method, gran = base.split('_')[1], base.split('_')[-1]
    batches = calib_batches(plane, cases, src, a.topk if method == 'maxslice' else 0)
    extra = {'WeightSymmetric': True, 'ActivationSymmetric': False}
    calib = CalibrationMethod.Percentile if method == 'percentile' else CalibrationMethod.MinMax
    if method == 'percentile':
        extra['CalibPercentile'] = a.percentile
    if method == 'entropy2048':
        extra['TensorQuantOverrides'] = entropy_ranges(pre, batches)
    quantize_static(str(pre), str(dst), Reader(batches), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=(gran == 'pc'),
                    reduce_range=rr, calibrate_method=calib, extra_options=extra,
                    nodes_to_exclude=[first_conv_name(pre)] if method == 'conv1fp32' else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--variants', nargs='+', default=[v for v in VARIANTS if v != 'fp32'])
    ap.add_argument('--calib-every', type=int, default=30, help='use every N-th internal-training exam')
    ap.add_argument('--topk', type=int, default=4, help='slices per exam for int8_maxslice_pc')
    ap.add_argument('--percentile', type=float, default=99.99)
    a = ap.parse_args()
    b = Path(a.bundle)
    _, itrain = split_cases('itrain')
    cases = itrain[::a.calib_every]
    rec_path = b / 'quant_record.json'
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {'variants': {}}
    rec.update(calib_cases=cases, ort=ort.__version__, time=time.strftime('%F %T'))
    (b / '_pre').mkdir(exist_ok=True)
    for p in PLANES:
        src, pre = b / 'fp32' / f'{p}.onnx', b / '_pre' / f'{p}.onnx'
        if not pre.exists():
            quant_pre_process(str(src), str(pre))
        for v in map(canonical, a.variants):
            dst = b / v / f'{p}.onnx'
            if dst.exists():
                continue
            dst.parent.mkdir(exist_ok=True)
            t0 = time.time()
            build(v, src, pre, dst, p, cases, a)
            rec['variants'].setdefault(v, {})[p] = {'bytes': dst.stat().st_size, 'sec': round(time.time() - t0, 1)}
            print(v, p, rec['variants'][v][p], flush=True)
    rec_path.write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
