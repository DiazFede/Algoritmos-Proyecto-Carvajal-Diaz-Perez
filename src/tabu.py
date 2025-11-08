
import numpy as np
from .evaluate import repair_and_score

def local_swap(x, rng):
    y = x.copy()
    i = int(rng.integers(0, len(x)))
    y[i] = 1 - y[i]
    return y, i

def tabu_improve(inst, x0, seed=0, iters=600, tabu_len=25, cov_tolerance=0.01, energy_tol=1e-3):
    """
    Lightweight tabu refinement: prioritize lowering energy while keeping coverage (neg_coverage)
    within a tolerance, as described in the hybrid NSGA-II + Tabú approach.
    """
    rng = np.random.default_rng(seed)
    xr, f = repair_and_score(x0, inst)
    f = np.array(f, dtype=float)
    bestx, bestf = xr.copy(), f.copy()
    tabu = []

    def coverage(obj):
        return -obj[0]

    for _ in range(iters):
        y, idx = local_swap(xr, rng)
        if idx in tabu:
            continue
        yr, g = repair_and_score(y, inst)
        g = np.array(g, dtype=float)

        curr_cov = coverage(f)
        cand_cov = coverage(g)
        coverage_ok = cand_cov >= (1.0 - cov_tolerance) * curr_cov
        energy_better = g[1] < f[1] - energy_tol

        if coverage_ok and energy_better:
            xr, f = yr, g
            if coverage(g) >= (1.0 - cov_tolerance) * coverage(bestf) and g[1] < bestf[1] - energy_tol:
                bestx, bestf = yr, g
            tabu.append(idx)
            if len(tabu) > tabu_len:
                tabu.pop(0)

    return bestx, bestf
