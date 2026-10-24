"""Resumable measurement queue for the Raspberry Pi 5.

Usage: python tools/pi_queue.py --home ~/mrnetq_pi --tasks configs/pi_tasks.json
"""
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def temp_c():
    try:
        return int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
    except OSError:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--home', default=str(Path.home() / 'mrnetq_pi'))
    ap.add_argument('--tasks', default=str(REPO / 'configs/pi_tasks.json'))
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--start-temp', type=float, default=60.0)
    ap.add_argument('--min-freq', type=float, default=2300.0)
    ap.add_argument('--idle-sec', type=int, default=60)
    a = ap.parse_args()
    H = Path(a.home)
    for d in ('logs', 'results'):
        (H / d).mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f'{time.strftime("%F %T")} {msg}'
        print(line, flush=True)
        with open(H / 'logs/queue_status.txt', 'a') as f:
            f.write(line + '\n')

    lock = open(H / 'logs/queue.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit('queue already running')
    env = dict(os.environ, PYTHONPATH=str(REPO))
    env.setdefault('MRNETQ_DATA', str(H / 'data/MRNet-v1.0'))
    pm = subprocess.Popen(['taskset', '-c', '0', a.python, str(REPO / 'tools/pmic_logger.py'),
                           '--out', str(H / 'logs/pmic.csv')], stderr=subprocess.DEVNULL)
    log(f'session start; PMIC logger pid {pm.pid}; idle baseline {a.idle_sec} s')
    t_idle = time.time()
    time.sleep(a.idle_sec)
    with open(H / 'logs/idle_windows.csv', 'a') as f:
        f.write(f'{t_idle:.3f},{time.time():.3f}\n')
    try:
        for task in json.loads(Path(a.tasks).read_text()):
            out = H / 'results' / task['id']
            rec = out / 'record.json'
            attempts = 0
            while True:
                if rec.exists() and json.loads(rec.read_text()).get('status') == 'done':
                    r = json.loads(rec.read_text())
                    if (r.get('freq_mhz_min') or 1e9) >= a.min_freq or task.get('quality_only'):
                        break
                    if attempts >= 3:
                        r['throttle_flag'] = True
                        rec.write_text(json.dumps(r, indent=1))
                        log(f'THROTTLE FLAG {task["id"]}: kept attempt {attempts} (min {r.get("freq_mhz_min")} MHz)')
                        break
                    log(f'throttled {task["id"]} (min {r.get("freq_mhz_min")} MHz), repeating')
                    shutil.rmtree(out)
                if attempts >= 3:
                    break
                while temp_c() > a.start_temp:
                    time.sleep(5)
                attempts += 1
                log(f'RUN {task["id"]} (attempt {attempts}, {temp_c():.1f} C)')
                if task.get('no_pmic'):
                    pm.send_signal(signal.SIGSTOP)
                p = subprocess.run([a.python, '-m', 'mrnetq.infer', '--bundle', str(H / 'models' / task['bundle']),
                                    '--out', str(out)] + task['args'], env=env, cwd=str(REPO),
                                   stdout=open(H / 'logs' / f'{task["id"]}.log', 'w'), stderr=subprocess.STDOUT)
                if task.get('no_pmic'):
                    pm.send_signal(signal.SIGCONT)
                if p.returncode != 0:
                    log(f'ERROR {task["id"]} rc={p.returncode}')
                    break
            if rec.exists():
                r = json.loads(rec.read_text())
                log(f'DONE {task["id"]}: median {r["t_total_median"]:.3f} s, RSS {r["peak_rss_mb"]} MB, '
                    f'min {r.get("freq_mhz_min")} MHz, max {r.get("temp_max")} C, cool-down {r.get("cooldown_sec")} s')
    finally:
        pm.terminate()
        log('queue finished')


if __name__ == '__main__':
    main()
