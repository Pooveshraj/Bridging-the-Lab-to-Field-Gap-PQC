"""
FRB-Sim core: leakage simulation of ML-KEM (Kyber) base multiplication on a
Cortex-M4-like device and non-profiled CPA, under laboratory and field conditions.

Target intermediate (pqm4 / PQClean reference, ref/reduce.c and ref/ntt.c):
    fqmul(a, b) = montgomery_reduce((int32)a * b)
    montgomery_reduce(a): t = (int16)(a * QINV); t = (a - (int32)t * KYBER_Q) >> 16
In basemul(r, a, b, zeta) the product fqmul(a[0], b[0]) depends on one secret
NTT-domain coefficient b[0] = s_hat[j] and one public coefficient a[0] = u_hat[j]
(derived from the ciphertext). Leakage model: Hamming weight of the 16-bit
two's-complement result. Divide-and-conquer: one coefficient, q = 3329 hypotheses.
"""
import numpy as np

Q = 3329
QINV = -3327  # q^-1 mod 2^16 (signed), as in the Kyber reference code


def _to_int16(x):
    x = np.asarray(x, dtype=np.int64) & 0xFFFF
    return np.where(x >= 0x8000, x - 0x10000, x)


def montgomery_reduce(a):
    a = np.asarray(a, dtype=np.int64)
    t = _to_int16(a * QINV)
    return (a - t * Q) >> 16


def fqmul(a, b):
    return montgomery_reduce(np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64))


def hw16(x):
    x = np.asarray(x, dtype=np.int64) & 0xFFFF
    return np.unpackbits(x.astype('>u2').view(np.uint8).reshape(-1, 2), axis=1).sum(1).reshape(np.shape(x))


_TABLE = None


def hw_table():
    """HW(fqmul(u, g)) for all public u in [0,q) and key guesses g in [0,q): uint8 [q, q]."""
    global _TABLE
    if _TABLE is None:
        u = np.arange(Q)[:, None]
        g = np.arange(Q)[None, :]
        _TABLE = hw16(fqmul(u, g)).astype(np.uint8)
    return _TABLE


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------
def gen_traces(rng, n, key, cfg):
    """Generate n simulated traces for one secret coefficient `key`.

    Returns (u, T): public inputs u [n] and traces T [n, L] (float32).

    cfg keys (all optional; defaults = laboratory tier):
      L        window length in leakage points (default 32)
      t0       position of the target intermediate in the lab (default 8)
      sigma    Gaussian noise std at the probe (lab calibration)
      gain     signal gain (F4 distance/enclosure), default 1.0
      gain_jit relative per-trace gain variation (F3 battery droop), default 0
      p_mis    probability that trace alignment fails (F1 RTOS preemption)
      jit_sd   std of residual jitter (leakage points) after alignment (F1)
      lp_alpha first-order low-pass coefficient (F3 regulator), 0 = off
      ripple   amplitude of regulator switching ripple (F3), 0 = off
      p_burst  probability a radio event overlaps the trace (F2)
      burst_k  burst noise std as a multiple of sigma (F2)
      shuffle  number of positions the target can occupy (countermeasure), 1 = off
    """
    L = cfg.get('L', 32); t0 = cfg.get('t0', 8)
    sigma = cfg['sigma']; gain = cfg.get('gain', 1.0)
    shuffle = cfg.get('shuffle', 1)

    u = rng.integers(0, Q, n)
    # Algorithmic noise: other intermediates of the loop are independent
    # fqmul outputs of random operands.
    other = hw16(fqmul(rng.integers(0, Q, (n, L)), rng.integers(0, Q, (n, L)))).astype(np.float32)
    target = hw_table()[u, key].astype(np.float32)

    # Position of the target in each trace.
    if shuffle > 1:
        pos = t0 + rng.integers(0, shuffle, n)
    else:
        pos = np.full(n, t0)
    p_mis = cfg.get('p_mis', 0.0); jit = cfg.get('jit_sd', 0.0)
    if jit > 0:
        pos = pos + np.rint(rng.normal(0, jit, n)).astype(int)
    if p_mis > 0:
        mis = rng.random(n) < p_mis
        pos = np.where(mis, rng.integers(0, L, n), pos)
    pos = np.clip(pos, 0, L - 1)
    sig = other
    sig[np.arange(n), pos] = target

    # Centre the HW signal (a DC offset does not change correlation).
    sig = sig - 8.0
    g = np.full((n, 1), gain, dtype=np.float32)
    gj = cfg.get('gain_jit', 0.0)
    if gj > 0:
        g = g * (1.0 - gj * rng.random((n, 1))).astype(np.float32)
    sig = sig * g

    a = cfg.get('lp_alpha', 0.0)
    if a > 0:  # first-order IIR low-pass: y_t = (1-a) x_t + a y_{t-1}
        y = np.empty_like(sig); y[:, 0] = sig[:, 0]
        for t in range(1, L):
            y[:, t] = (1 - a) * sig[:, t] + a * y[:, t - 1]
        sig = y
    rp = cfg.get('ripple', 0.0)
    if rp > 0:
        ph = rng.uniform(0, 2 * np.pi, (n, 1))
        per = cfg.get('ripple_period', 7.3)
        sig = sig + rp * np.sin(2 * np.pi * np.arange(L)[None, :] / per + ph)

    noise = rng.normal(0, sigma, (n, L)).astype(np.float32)
    pb = cfg.get('p_burst', 0.0)
    if pb > 0:
        b = rng.random(n) < pb
        noise[b] *= cfg.get('burst_k', 5.0)
    return u, (sig + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Attacker preprocessing
# ---------------------------------------------------------------------------
def preprocess(T, cfg):
    """Attacker-side processing.
    integ  : moving-window integration width (counters jitter), 1 = off
    reject : drop traces whose variance exceeds reject * median variance (radio bursts)
    sumall : sum all samples (classic attack on shuffled implementations)
    Returns processed traces and a boolean mask of kept traces."""
    keep = np.ones(len(T), bool)
    rj = cfg.get('reject', 0.0)
    if rj > 0:
        v = T.var(1)
        keep = v <= rj * np.median(v)
    w = cfg.get('integ', 1)
    if w > 1:
        k = np.ones(w, np.float32)
        T = np.apply_along_axis(lambda r: np.convolve(r, k, 'valid'), 1, T)
    if cfg.get('sumall', False):
        T = T.sum(1, keepdims=True)
    return T, keep


# ---------------------------------------------------------------------------
# CPA with incremental statistics, evaluated at a grid of trace counts
# ---------------------------------------------------------------------------
def cpa_success_curve(rng, key, cfg, grid, chunk=2000, gen=None, prep=None):
    gen = gen or gen_traces; prep = prep or preprocess
    """Run one attack experiment. Returns rank of the correct key at each
    trace count in `grid` (traces acquired, including rejected ones)."""
    H = hw_table()
    G = Q
    nmax = grid[-1]
    sH = np.zeros(G); sH2 = np.zeros(G)
    sT = None; sT2 = None; sHT = None; n_used = 0
    ranks = []; gi = 0; acquired = 0
    while acquired < nmax:
        m = min(chunk, grid[gi] - acquired)
        u, T = gen(rng, m, key, cfg)
        T, keep = prep(T, cfg)
        u = u[keep]; T = T[keep].astype(np.float64)
        Hc = H[u].astype(np.float64)                     # [m, G]
        if sT is None:
            Lp = T.shape[1]
            sT = np.zeros(Lp); sT2 = np.zeros(Lp); sHT = np.zeros((G, Lp))
        sH += Hc.sum(0); sH2 += (Hc ** 2).sum(0)
        sT += T.sum(0); sT2 += (T ** 2).sum(0)
        sHT += Hc.T @ T
        n_used += len(u); acquired += m
        if acquired == grid[gi]:
            n = max(n_used, 2)
            num = n * sHT - np.outer(sH, sT)
            den = np.sqrt(np.maximum(n * sH2 - sH ** 2, 1e-12))[:, None] * \
                  np.sqrt(np.maximum(n * sT2 - sT ** 2, 1e-12))[None, :]
            rho = np.nan_to_num(np.abs(num / den), nan=0.0).max(1)
            ranks.append(int((rho > rho[key]).sum()) + 1)
            gi += 1
    return np.array(ranks)


def success_rate(rng, cfg, grid, reps, gen=None, prep=None):
    """Success rate (rank 1) at each grid point over `reps` independent keys."""
    R = np.zeros((reps, len(grid)), int)
    for r in range(reps):
        key = int(rng.integers(1, Q))  # key 0 gives a constant model (no leakage)
        R[r] = cpa_success_curve(rng, key, cfg, grid, gen=gen, prep=prep)
    return (R == 1).mean(0), R


def n_at(sr, grid, level=0.9):
    """Smallest trace count reaching success rate `level` (log-linear
    interpolation between grid points). Returns inf if never reached."""
    grid = np.asarray(grid, float)
    idx = np.where(sr >= level)[0]
    if len(idx) == 0:
        return np.inf
    i = idx[0]
    if i == 0:
        return grid[0]
    x0, x1 = np.log(grid[i - 1]), np.log(grid[i]); y0, y1 = sr[i - 1], sr[i]
    if y1 == y0:
        return grid[i]
    return float(np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))


def bootstrap_n(R, grid, level=0.9, B=1000, seed=0):
    """Percentile bootstrap CI for n_at over repetitions."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        s = R[rng.integers(0, len(R), len(R))]
        vals.append(n_at((s == 1).mean(0), grid, level))
    vals = np.array(vals)
    finite = vals[np.isfinite(vals)]
    lo = np.percentile(finite, 2.5) if len(finite) else np.inf
    hi = np.percentile(vals, 97.5) if np.isfinite(vals).mean() > 0.975 else np.inf
    return lo, hi
