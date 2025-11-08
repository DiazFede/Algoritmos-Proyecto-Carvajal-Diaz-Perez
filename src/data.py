
from dataclasses import dataclass
import numpy as np

@dataclass
class Satellite:
    sid: int
    e_max: float      # energy budget per planning horizon
    t_max: float      # max operable time
    slew_coeff: float # energy per rad of slew

@dataclass
class Target:
    tid: int
    priority: float    # w_t
    critical: bool
    min_obs: int

@dataclass
class Opportunity:
    oid: int
    sat: int          # satellite id
    tid: int          # target id
    start: float
    end: float
    duration: float
    energy: float
    theta: float      # pointing angle needed before this obs (proxy)
    cov: float        # informational value (base)
    p_cloud: float    # cloud probability (reduces expected value)

@dataclass
class Instance:
    sats: list
    targets: list
    ops: list
    horizon: float
    max_expected_coverage: float

def gen_instance(n_sat=5, n_targets=60, n_ops=800, seed=0):
    """
    Synthetic instance aligned with the critical-event observation report.
    Default configuration: 5 satellites, 60 targets, 800 observation opportunities.
    """
    rng = np.random.default_rng(seed)
    sats = []
    for s in range(n_sat):
        e_max = rng.uniform(80, 140)
        t_max = rng.uniform(120, 200)
        slew_coeff = rng.uniform(0.3, 0.8)
        sats.append(Satellite(s, e_max, t_max, slew_coeff))

    targets = []
    for t in range(n_targets):
        priority = rng.choice([1,2,3,4,5], p=[0.15,0.25,0.3,0.2,0.1])
        critical = (priority >= 4 and rng.random() < 0.6)
        min_obs = 1 if critical else 0
        targets.append(Target(t, float(priority), bool(critical), min_obs))

    # timeline horizon
    H = 600.0
    ops = []
    total_expected = 0.0
    for o in range(n_ops):
        sat = int(rng.integers(0, n_sat))
        tid = int(rng.integers(0, n_targets))
        start = float(rng.uniform(0, H-30))
        duration = float(rng.uniform(5, 30))
        end = start + duration
        energy = float(rng.uniform(1.0, 5.0) * (1.0 + duration/40))
        theta = float(abs(rng.normal(0.6, 0.3)))
        cov = float(rng.uniform(0.5, 2.5))
        p_cloud = float(np.clip(rng.beta(2,5), 0, 0.9))  # more clear than cloudy
        total_expected += targets[tid].priority * cov * (1.0 - p_cloud)
        ops.append(Opportunity(o, sat, tid, start, end, duration, energy, theta, cov, p_cloud))

    return Instance(sats, targets, ops, H, total_expected)
