"""
FRB-Sim trace model: every model component and parameter is taken from the paper's reference list.

Device and sampling  : Kyber-768 on a Cortex-M4 (STM32F405) at 24 MHz, sampled at four
                       times the clock (x4), window of 480 samples (the Decode dataset) [Huang2024].
Target and leakage   : base multiplication fqmul(u_hat, s_hat) with Hamming-weight leakage
                       and Gaussian noise, CPA on polynomial multiplication [Assael2023, Mujdei2024, Tosun2024, Chen2021].
Laboratory tier      : triggered and synchronously clocked [Mujdei2024, Xu2022]; noise calibrated so
                       that N90 ~ 150 traces [Tosun2024] (157 per coefficient in [Chen2021]).
C1 random delay      : shifting deformation SH_T, window start uniform in [0, T] samples,
                       T in {5, 10, 15, 20, 50} [Huang2024]; DASH-T random delays [He2025].
C2 clock jitter      : add-remove deformation AR_R, R points inserted and R removed per
                       trace on top of base jitter R0 ~ N(0, s2), s2 = 4 (low) or 36 (high),
                       R in {0, 20, 50, 100, 200} [Huang2024]; RDFS [Wang2026].
C3 noise             : Gaussian noise added at SNR in {5, 3, 1, 0.1} [Huang2024]; equivalent to
                       variances 0.1-10 on normalised traces [Chen2026]; artificial noise [Liu2021].
C4 low sampling rate : one sample per clock (x1) instead of x4 [Huang2024]; 1 vs 10 GS/s [Tosun2024].
Attacker             : sums the samples of one clock period (sequential points combined [Tosun2024]).
Countermeasure       : shuffling of N = 16 operations [Park2025]; shuffling plus random delay of
                       0-15 clock cycles as in DASH-T [He2025].
"""
import numpy as np
from .core import Q, fqmul, hw16, hw_table

SPO = 4             # samples per clock (x4 sampling, [Huang2024])
L = 480             # samples in the analysed window (Decode dataset, [Huang2024])
CLK = L // SPO      # 120 clock cycles, one intermediate per cycle
T_CLK = CLK // 2    # target in the middle of the window
HW_VAR = 10.34      # variance of HW(fqmul) outputs


def gen_traces(rng, n, key, cfg):
    sigma = cfg['sigma']
    T = int(cfg.get('delay_T', 0)) + int(cfg.get('cm_delay', 0)) * SPO
    R = int(cfg.get('jitter_R', 0)); s2 = cfg.get('jitter_s2', 0.0)
    shuffle = cfg.get('shuffle', 1)
    pre = int(np.ceil(T / SPO)) + 2
    jpad = int(np.ceil((R + 3 * np.sqrt(s2)) / SPO)) + 2
    nclk = pre + CLK + jpad

    u = rng.integers(0, Q, n)
    ops = hw16(fqmul(rng.integers(0, Q, (n, nclk)), rng.integers(0, Q, (n, nclk)))).astype(np.float32)
    tgt = hw_table()[u, key].astype(np.float32)
    pos = pre + T_CLK + (rng.integers(0, shuffle, n) - shuffle // 2 if shuffle > 1 else 0)
    ops[np.arange(n), pos] = tgt
    ops -= 8.0
    sig = np.repeat(ops, SPO, axis=1)
    Lb = sig.shape[1]

    start = np.full((n, 1), pre * SPO)
    if T > 0:                                   # C1 SH_T: random start in [0, T]
        start = start - rng.integers(0, T + 1, (n, 1))
    idx = start + np.arange(L)[None, :]
    if R > 0 or s2 > 0:                         # C2 AR_R on top of base jitter N(0, s2)
        Rt = R + (np.abs(rng.normal(0, np.sqrt(s2), (n, 1))) if s2 > 0 else 0.0)
        p = np.clip(np.asarray(Rt, float) / L, 0, 0.95)
        off = np.cumsum(rng.random((n, L)) < p, 1) - np.cumsum(rng.random((n, L)) < p, 1)
        idx = idx + off
    idx = np.clip(idx, 0, Lb - 1)
    X = np.take_along_axis(sig, idx, 1)
    if cfg.get('low_rate', False):              # C4 x1 sampling: one sample per clock
        X = X[:, ::SPO]
    noise = rng.normal(0, sigma, X.shape).astype(np.float32)
    snr = cfg.get('snr', None)
    if snr:                                     # C3 added noise at SNR = trace power / noise power
        noise += rng.normal(0, np.sqrt((HW_VAR + sigma ** 2) / snr), X.shape).astype(np.float32)
    return u, (X + noise).astype(np.float32)


def preprocess(X, cfg):
    keep = np.ones(len(X), bool)
    wi = cfg.get('integ', 0)
    if wi > 1:  # sliding integration over wi samples (stride one clock) against desynchronisation
        c = np.cumsum(np.pad(X, ((0, 0), (1, 0))), 1)
        st = 1 if cfg.get('integ_x1') else SPO
        return (c[:, wi:] - c[:, :-wi])[:, ::st], keep
    w = cfg.get('combine', 1)
    if w > 1:   # sum each clock period (non-overlapping bins of w samples) [Tosun2024]
        n, m = X.shape
        X = X[:, :m - m % w].reshape(n, -1, w).sum(2)
    return X, keep
