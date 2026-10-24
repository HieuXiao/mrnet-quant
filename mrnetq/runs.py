"""Run names shared by make_tasks, pi_queue and make_tables.

quality: <backbone>_s<seed>_<split>_<variant>_k<K>
latency: lat_<variant>_k<K>_t<threads>
logger overhead: overhead_<variant>_pmic_<on|off>
"""


def quality(backbone, seed, split, variant, k=0):
    return f'{backbone}_s{seed}_{split}_{variant}_k{k}'


def latency(variant, k=0, threads=4):
    return f'lat_{variant}_k{k}_t{threads}'


def overhead(variant, on):
    return f'overhead_{variant}_pmic_{"on" if on else "off"}'
