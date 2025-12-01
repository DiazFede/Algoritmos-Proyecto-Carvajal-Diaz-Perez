
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.population import Population
from .greedy import greedy_solution

from .data import gen_instance
from .evaluate import evaluate_vector, repair_and_score

class SatProblem(ElementwiseProblem):
    """Problema de optimización binaria para planificación satelital."""

    def __init__(self, inst):
        """
        Args:
            inst (Instance): Instancia con oportunidades y parámetros del problema.
        """
        super().__init__(n_var=len(inst.ops),
                         n_obj=3,
                         n_constr=0,
                         xl=0, xu=1, type_var=np.bool_)
        self.inst = inst

    def _evaluate(self, x, out, *args, **kwargs):
        """Evalúa un individuo binario y devuelve objetivos reparados."""
        objs = evaluate_vector(x, self.inst)
        out["F"] = np.array(objs, dtype=float)

def run_nsga2(n_sat=5, n_targets=60, n_ops=800, seed=0,
              pop_size=200, n_gen=300, p_c=0.8, p_m=0.1, inst=None):
    """
    Ejecuta NSGA-II sobre el modelo satelital.

    Args:
        n_sat (int): Número de satélites a generar si no se pasa instancia.
        n_targets (int): Número de objetivos a generar si no se pasa instancia.
        n_ops (int): Número de oportunidades a generar si no se pasa instancia.
        seed (int): Semilla de aleatoriedad.
        pop_size (int): Tamaño de población.
        n_gen (int): Número de generaciones.
        p_c (float): Probabilidad de crossover HUX.
        p_m (float): Probabilidad de mutación bitflip.
        inst (Instance|None): Instancia precalculada para alinear experimentos.

    Returns:
        tuple: (instancia, resultado pymoo, soluciones reparadas, objetivos reparados)
    """
    if inst is None:
        inst = gen_instance(n_sat, n_targets, n_ops, seed=seed)

    problem = SatProblem(inst)

   
    # --- sampling: inject Greedy + random ---
    greedy_x = greedy_solution(inst, seed).astype(bool)
    rnd = BinaryRandomSampling().do(problem, pop_size-1)
    X0 = np.vstack([greedy_x, rnd.get("X")])

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=Population.new("X", X0),  # <<< usar X0 como población inicial
        crossover=HUX(prob=p_c),
        mutation=BitflipMutation(prob=p_m),
        eliminate_duplicates=True
    )



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
