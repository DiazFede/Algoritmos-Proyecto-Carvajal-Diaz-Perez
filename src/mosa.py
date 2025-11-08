
import numpy as np
from .evaluate import repair_and_score

def neighbor(x, rng):
    """
    Generate a neighbor by either flipping a sparse subset of decisions
    or swapping two indices (swap move keeps the number of active obs).
    """
    y = x.copy()
    if rng.random() < 0.5:
        k = rng.integers(1, max(2, len(x)//20))
        idx = rng.choice(len(x), size=k, replace=False)
        y[idx] = 1 - y[idx]
    else:
        i, j = rng.choice(len(x), size=2, replace=False)
        if y[i] != y[j]:
            y[i], y[j] = y[j], y[i]
        else:
            y[i] = 1 - y[i]
            y[j] = 1 - y[j]
    return y

def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

def run_mosa(inst, seed=0, iters=3000, T0=2.0, alpha=0.997):
    """
    Multiobjective simulated annealing (MOSA) per report:
    - swap/flip neighborhood
    - gradual cooling with alpha≈0.997
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0,2,size=len(inst.ops)).astype(int)
    xr, f = repair_and_score(x, inst)

    pareto = [ (xr, np.array(f)) ]

    T = T0
    for it in range(iters):
        y = neighbor(xr, rng)
        yr, g = repair_and_score(y, inst)
        g = np.array(g, dtype=float)

        if dominates(g, f) or np.allclose(g, f):
            xr, f = yr, g
            pareto.append( (xr, g) )
        else:
            # random scalarization as acceptance key
            w = rng.random(3); w = w/np.sum(w)
            sf = float(np.dot(w, f))
            sg = float(np.dot(w, g))
            if sg < sf:
                xr, f = yr, g
            else:
                prob = np.exp(-(sg - sf)/max(1e-9, T))
                if rng.random() < prob:
                    xr, f = yr, g
        T *= alpha

    Fs = np.array([p[1] for p in pareto])
    keep = []
    for i in range(len(Fs)):
        dominated = False
        for j in range(len(Fs)):
            if i!=j and np.all(Fs[j] <= Fs[i]) and np.any(Fs[j] < Fs[i]):
                dominated = True; break
        if not dominated:
            keep.append(i)
    sols = [pareto[i][0] for i in keep]
    Fs_keep = Fs[keep]
    return sols, Fs_keep
