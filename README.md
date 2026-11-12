# FRB-Sim

Simulation code for Section XI of the paper

> P. R. K and S. Vairavasundaram, "Bridging the Lab-to-Field Gap in Physical Security of Lattice-Based
> Post-Quantum Cryptography for IoT Devices: A Systematic Review," manuscript under review, 2026.

## Overview

FRB-Sim estimates how much harder a side-channel attack on ML-KEM (CRYSTALS-Kyber) becomes when it moves
from a laboratory set-up to a deployed IoT device. It is the simulation part of the Field-Realistic
Benchmarking (FRB) framework proposed in the paper.

* **Target.** The ML-KEM (Kyber-768) base multiplication `fqmul(a, b) = montgomery_reduce(a * b)`, q = 3329,
  as in the Kyber reference code. One secret NTT-domain coefficient is multiplied by a public coefficient
  taken from the ciphertext.
* **Leakage model.** Hamming weight of the 16-bit result plus Gaussian noise, on a Cortex-M4-class device
  with four samples per clock and a 480-sample window.
* **Attack.** Non-profiled correlation power analysis (CPA) over all 3,329 hypotheses of the secret
  coefficient.
* **Metric.** N90, the number of traces at which the key-recovery success rate reaches 90%, with a 95%
  percentile-bootstrap confidence interval over the repetitions. The realism degradation factor of a
  condition is its N90 divided by the laboratory N90.

## Simulated conditions

| Group | Conditions |
| --- | --- |
| Laboratory | synchronised traces, noise calibrated to an N90 of about 150 traces |
| C1 random delay | shifting deformation SH_T, T = 5, 10, 20, 50 samples |
| C2 clock jitter | add-remove deformation AR_R, R = 20, 50, 100, 200, base jitter variance 36 |
| C3 added noise | SNR = 5, 3, 1, 0.1 |
| C4 sampling rate | one sample per clock instead of four |
| Edge tier | SH10, low jitter (variance 4) and SNR 5 |
| Field tier | SH50, AR200 (variance 36), SNR 1 and one sample per clock |
| Countermeasures | shuffling of 16 operations, and shuffling with a random delay of 0-15 clock cycles |

The attacker adapts to each condition by summing the samples of each clock period and, for desynchronised
traces, integrating over the expected spread of the target. The source of every parameter is given in the
docstring of `frbsim/leakage.py` and in Table 19 of the paper.

## Repository structure

```
FRB-Sim/
├── frbsim/
│   ├── __init__.py
│   ├── core.py          Kyber fqmul and Montgomery reduction, CPA, success rate, N90, bootstrap CI
│   ├── leakage.py       trace model, conditions C1-C4 and countermeasures
│   └── experiments.py   definitions of all 18 conditions and the parallel experiment runner
├── README.md
└── LICENSE
```

## Requirements

* Python 3.10 or later
* NumPy (tested with 2.1.3)

```bash
pip install numpy==2.1.3
```

## Usage

From the repository folder:

```bash
python -m frbsim.experiments <output.json> [repetitions] [workers]
```

| Argument | Meaning | Default |
| --- | --- | --- |
| `output.json` | results file to write | required |
| `repetitions` | independent experiments per condition, each with a new random key | 100 |
| `workers` | parallel processes | all CPU cores |

Example with the settings used in the paper:

```bash
python -m frbsim.experiments results.json 500 32
```

Each condition is printed as it finishes, for example

```
[cond] Lab: N90=162 CI=[150.48..., 170.16...] (...s)
```

and the results file is updated after every condition. The laboratory noise level can be changed with the
environment variable `FRB_SIGMA_LAB` (default 12.0).

**Run time.** The 500-repetition run in the paper took about 1.5 hours on a 32-core server. The field tier
needs up to 300,000 traces per experiment and is the slowest job.

**Reproducibility.** The random seeds are fixed (base seed 4040, one seed per condition), so a run with
500 repetitions and the same NumPy version gives the N90 values and confidence intervals reported in the paper.

## Output format

`output.json` contains `sigma_lab`, `reps` and `results`. `results` groups the conditions under `cond`
(laboratory, C4, edge and field tiers), `C1 random delay SH_T`, `C2 clock jitter AR_R`, `C3 noise SNR` and
`counter` (countermeasures). Each entry holds

| Field | Meaning |
| --- | --- |
| `name` | condition name |
| `cfg` | simulation and attacker settings |
| `grid` | trace counts at which the attack was evaluated |
| `sr` | success rate at each trace count |
| `ranks` | rank of the correct key in every repetition at each trace count |
| `n90` | traces for a 90% success rate |
| `ci` | 95% bootstrap confidence interval of `n90` |
| `reps`, `nmax`, `secs` | repetitions, largest trace count and run time in seconds |

## Limitations

The traces are generated from an algorithm-level Hamming-weight model, not measured on hardware. The results
show how deployment conditions scale the cost of the attack relative to the laboratory. They are not absolute
trace counts for a particular device.

## Citation

If you use this code, please cite the paper above.

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE).
