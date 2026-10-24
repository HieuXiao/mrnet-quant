"""Log Raspberry Pi 5 board power from the PMIC (vcgencmd pmic_read_adc) to CSV.

Usage: taskset -c 0 python3 tools/pmic_logger.py --out logs/pmic.csv
"""
import argparse
import re
import signal
import subprocess
import sys
import time

PAT = re.compile(r'^\s*(\S+)_([AV])\s+(?:current|volt)\(\d+\)=([\d.]+)[AV]')


def read_power():
    out = subprocess.run(['vcgencmd', 'pmic_read_adc'], capture_output=True, text=True, timeout=2).stdout
    amps, volts = {}, {}
    for line in out.splitlines():
        m = PAT.match(line)
        if m:
            (amps if m.group(2) == 'A' else volts)[m.group(1)] = float(m.group(3))
    return sum(amps[r] * volts[r] for r in amps if r in volts), len(set(amps) & set(volts))


def read_temp():
    try:
        return int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
    except OSError:
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--interval', type=float, default=0.1)
    a = ap.parse_args()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    w, n = read_power()
    print(f'{n} PMIC rails, current power {w:.2f} W', file=sys.stderr, flush=True)
    with open(a.out, 'a') as f:
        if f.tell() == 0:
            f.write('t,watts,temp_c\n')
        while not stop:
            t = time.time()
            try:
                w, _ = read_power()
            except (OSError, subprocess.TimeoutExpired):
                continue
            f.write(f'{t:.4f},{w:.4f},{read_temp():.1f}\n')
            f.flush()
            time.sleep(max(0.0, a.interval - (time.time() - t)))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
