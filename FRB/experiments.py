"""FRB-Sim experiments: laboratory, conditions C1-C4, edge and field tiers and countermeasures.
Every setting is traced to the paper's reference list (see leakage.py).

Usage: python -m frbsim.experiments OUT.json [repetitions] [workers]
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from .core import success_rate, n_at, bootstrap_n
from . import leakage as core2

SIGMA_LAB = float(os.environ.get('FRB_SIGMA_LAB', 12.0))
SPO = core2.SPO


def cond(**kw):
    """Device-side settings from [Huang2024]; attacker adapts as in [Tosun2024]: it sums each clock period and,
    when traces are desynchronised, integrates over the expected spread of the target."""
    d = dict(sigma=SIGMA_LAB)
    d.update(kw)
    T = d.get('delay_T', 0) + d.get('cm_delay', 0) * SPO
    R = d.get('jitter_R', 0) + np.sqrt(d.get('jitter_s2', 0.0))
    spread = T + 2 * int(np.ceil(np.sqrt(2 * (R / core2.L) * (core2.L / 2)))) if R > 0 else T
    if d.get('shuffle', 1) > 1:
        spread += d['shuffle'] * SPO
    if d.get('low_rate'):
        d['combine'] = 1
        if spread > 0:
            d['integ'] = max(1, int(np.ceil(spread / SPO)) + 1) * 1  # x1 traces: one sample per clock
            d['integ_x1'] = True
    elif spread > 0:
        d['integ'] = SPO + spread
    else:
        d['combine'] = SPO
    return d


CONDITIONS = {
    'Lab':                              (cond(), 3_000),
    'C4 Low sampling rate x1':     (cond(low_rate=True), 20_000),
    'Edge tier':                        (cond(delay_T=10, jitter_s2=4, snr=5), 30_000),
    'Field tier':                       (cond(delay_T=50, jitter_R=200, jitter_s2=36, snr=1, low_rate=True), 300_000),
}

SWEEPS = {
    'C1 random delay SH_T': [(f'T={t}', cond(delay_T=t), 30_000) for t in (5, 10, 20, 50)],
    'C2 clock jitter AR_R': [(f'R={r}', cond(jitter_R=r, jitter_s2=36), 30_000) for r in (20, 50, 100, 200)],
    'C3 noise SNR':   [(f'SNR={s}', cond(snr=s), 30_000) for s in (5, 3, 1, 0.1)],
}

COUNTER = {
    'Lab, shuffling N=16':                  (cond(shuffle=16), 50_000),
    'Lab, shuffling + random delay 0-15':  (cond(shuffle=16, cm_delay=15), 100_000),
}


def grid_for(nmax, n0=10, per_decade=12):
    k = int(np.ceil(np.log10(nmax / n0) * per_decade))
    g = np.unique(np.round(n0 * 10 ** (np.arange(k + 1) / per_decade)).astype(int))
    g = g[g <= nmax]
    if g[-1] != nmax:
        g = np.append(g, nmax)
    return [int(x) for x in g]


def _run(args):
    name, cfg, nmax, reps, seed = args
    rng = np.random.default_rng(seed)
    grid = grid_for(nmax)
    t = time.time()
    sr, R = success_rate(rng, cfg, grid, reps, core2.gen_traces, core2.preprocess)
    n90 = n_at(sr, grid, 0.9)
    lo, hi = bootstrap_n(R, grid, 0.9, B=500, seed=seed)
    return dict(name=name, cfg=cfg, nmax=nmax, reps=reps, grid=grid, sr=sr.tolist(),
                ranks=R.tolist(), n90=n90, ci=[lo, hi], secs=time.time() - t)


def run_all(out, reps=100, workers=2, seed=4040):
    jobs = []; i = 0
    for n, (c, m) in CONDITIONS.items():
        jobs.append(('cond', (n, c, m, reps, seed + i))); i += 1
    for s, lst in SWEEPS.items():
        for n, c, m in lst:
            jobs.append((s, (n, c, m, reps, seed + i))); i += 1
    for n, (c, m) in COUNTER.items():
        jobs.append(('counter', (n, c, m, reps, seed + i))); i += 1
    jobs.sort(key=lambda j: -j[1][2])
    out_d = {}
    with Pool(workers) as p:
        for (grp, _), r in zip(jobs, p.imap(_run, [j[1] for j in jobs])):
            out_d.setdefault(grp, []).append(r)
            print(f"[{grp}] {r['name']}: N90={r['n90']:.0f} CI={r['ci']} ({r['secs']:.0f}s)", flush=True)
            json.dump(dict(sigma_lab=SIGMA_LAB, reps=reps, results=out_d), open(out, 'w'))
    return out_d


if __name__ == '__main__':
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    run_all(sys.argv[1], reps=reps, workers=workers)
