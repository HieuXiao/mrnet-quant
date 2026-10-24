"""Write configs/pi_tasks.json: quality runs first, then latency and energy runs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrnetq.runs import latency, overhead, quality  # noqa: E402

CALIB = ('int8_minmax_pc', 'int8_minmax_pt', 'int8_percentile_pc', 'int8_entropy2048_pc', 'int8_maxslice_pc',
         'int8_conv1fp32_pc')
H4 = ('fp32', 'int8_minmax_pc', 'int8_percentile_pc', 'int8_entropy2048_pc')


def task(tid, bundle, variant, split='test', k=0, threads=4, repeats=1, warmup=5, quality_only=True, extra=()):
    args = ['--variant', variant, '--split', split, '--k', str(k), '--threads', str(threads), '--repeats',
            str(repeats), '--warmup', str(warmup)]
    if quality_only:
        args += ['--cool-above', '95']
    t = {'id': tid, 'bundle': bundle, 'args': args + list(extra)}
    if quality_only:
        t['quality_only'] = True
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'configs/pi_tasks.json'))
    a = ap.parse_args()
    T = []
    for s in (0, 1, 2):
        b = f'alexnet_seed{s}'
        for v in ('fp32', 'fp16') + CALIB:
            extra = ['--save-feats'] if s == 0 and v in H4 else []
            T.append(task(quality('alexnet', s, 'test', v), b, v, extra=extra))
        for v in ('fp32', 'fp16') + CALIB:
            T.append(task(quality('alexnet', s, 'ival', v), b, v, split='ival'))
        for v in ('fp32',) + CALIB:
            T.append(task(quality('mobilenet_v3_small', s, 'test', v), f'mobilenet_v3_small_seed{s}', v))
    for k in (24, 16, 8):
        for v in ('fp32', 'fp16', 'int8_minmax_pc'):
            T.append(task(quality('alexnet', 0, 'test', v, k), 'alexnet_seed0', v, k=k))
    T.append(task(quality('alexnet', 0, 'test', 'int8_minmax_pc_rr'), 'alexnet_seed0', 'int8_minmax_pc_rr'))
    for k in (0, 24, 16, 8):
        for threads in (4, 1):
            for v in ('fp32', 'fp16', 'int8_minmax_pc'):
                T.append(task(latency(v, k, threads), 'alexnet_seed0', v, k=k, threads=threads, repeats=3, warmup=10,
                              quality_only=False))
    T.append(task(latency('int8_minmax_pc_rr'), 'alexnet_seed0', 'int8_minmax_pc_rr', repeats=3, warmup=10,
                  quality_only=False))
    for v in ('fp32', 'int8_minmax_pc'):
        for on in (True, False):
            t = task(overhead(v, on), 'alexnet_seed0', v, repeats=3, warmup=10, quality_only=False)
            if not on:
                t['no_pmic'] = True
            T.append(t)
    Path(a.out).write_text(json.dumps(T, indent=1))
    print(len(T), 'tasks ->', a.out)


if __name__ == '__main__':
    main()
