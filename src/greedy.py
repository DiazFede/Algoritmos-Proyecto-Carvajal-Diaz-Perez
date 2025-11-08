
import numpy as np
from .data import gen_instance
from .evaluate import repair_and_score, expected_value, evaluate_vector

def greedy_solution(inst, seed=0):
    rng = np.random.default_rng(seed)
    scores = []
    for op in inst.ops:
        v = expected_value(op, inst.targets)
        de = max(op.energy, 1e-6)
        scores.append((v/de, op.oid))
    scores.sort(reverse=True)
    x = np.zeros(len(inst.ops), dtype=int)
    for _, oid in scores:
        x[oid] = 1
        x_rep, _ = repair_and_score(x, inst)
        x = x_rep
    return x

def run_baseline(n_sat=5, n_targets=60, n_ops=800, seed=0, inst=None):
    """
    Greedy value/energy selector used as baseline.
    Allows reusing a provided instance to keep experiments aligned across algorithms.
    """
    if inst is None:
        inst = gen_instance(n_sat, n_targets, n_ops, seed)
    x = greedy_solution(inst, seed)
    objs = evaluate_vector(x, inst)
    _, _, meta = repair_and_score(x, inst, return_meta=True)
    return inst, x, objs, meta
