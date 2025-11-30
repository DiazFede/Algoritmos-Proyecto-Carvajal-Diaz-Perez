
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from .data import gen_instance
from .evaluate import evaluate_vector, repair_and_score

class SatProblem(ElementwiseProblem):
    def __init__(self, inst):
        super().__init__(n_var=len(inst.ops),
                         n_obj=3,
                         n_constr=0,
                         xl=0, xu=1, type_var=np.bool_)
        self.inst = inst

    def _evaluate(self, x, out, *args, **kwargs):
        objs = evaluate_vector(x, self.inst)
        out["F"] = np.array(objs, dtype=float)

def run_nsga2(n_sat=5, n_targets=60, n_ops=800, seed=0,
              pop_size=200, n_gen=30, p_c=0.8, p_m=0.1, inst=None):
    """
    NSGA-II configuration aligned with the report:
    - population 200, 300 generations
    - crossover prob 0.8 (HUX) and mutation 0.1
    Optionally accepts a pre-built instance to keep experiments synchronized.
    """
    if inst is None:
        inst = gen_instance(n_sat, n_targets, n_ops, seed=seed)

    problem = SatProblem(inst)
    algorithm = NSGA2(pop_size=pop_size,
                      sampling=BinaryRandomSampling(),
                      crossover=HUX(prob=p_c),
                      mutation=BitflipMutation(prob=p_m),
                      eliminate_duplicates=True)

    termination = get_termination("n_gen", n_gen)

    res = minimize(problem, algorithm, termination,
                   seed=seed, save_history=False, verbose=False)

    # Repair final pop to feasible selections & compute repaired objectives
    repaired = []
    F_rep = []
    for x in res.X:
        xr, objs = repair_and_score(x.astype(int), inst)
        repaired.append(xr)
        F_rep.append(objs)

    return inst, res, np.array(repaired), np.array(F_rep)
