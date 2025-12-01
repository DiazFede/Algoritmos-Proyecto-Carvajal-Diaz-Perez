import numpy as np
from .data import gen_instance
from .evaluate import repair_and_score, expected_value, evaluate_vector

def greedy_solution(inst, seed=0):
    """
    Construye una solución greedy por densidad valor/energía sin reoptimizar.

    Args:
        inst (Instance): Instancia de planificación.
        seed (int): Semilla de aleatoriedad.

    Returns:
        np.ndarray: Vector binario de oportunidades seleccionadas.
    """
    rng = np.random.default_rng(seed)

    # Puntaje densidad clásico
    scored = []
    for op in inst.ops:
        v = expected_value(op, inst.targets)      # w_t * cov (sin nubes)
        de = max(op.energy, 1e-9)
        scored.append((v / de, op))

    scored.sort(key=lambda t: t[0], reverse=True)

    # Estado por satélite
    sat_energy = {s.sid: 0.0 for s in inst.sats}
    sat_time   = {s.sid: 0.0 for s in inst.sats}
    sat_budgetE = {s.sid: s.e_max for s in inst.sats}
    sat_budgetT = {s.sid: s.t_max for s in inst.sats}
    timelines = {s.sid: [] for s in inst.sats}     # lista de (start,end) aceptados

    x = np.zeros(len(inst.ops), dtype=int)

    # Selección greedy: un solo pase
    for _, op in scored:
        s = op.sat

        # check solape local en ese satélite
        feasible_time = True
        for (st,en) in timelines[s]:
            if not (op.end <= st or op.start >= en):
                feasible_time = False
                break
        if not feasible_time:
            continue

        # presupuestos locales
        if sat_energy[s] + op.energy > sat_budgetE[s]:
            continue
        if sat_time[s]   + op.duration > sat_budgetT[s]:
            continue

        # aceptar
        x[op.oid] = 1
        timelines[s].append((op.start, op.end))
        sat_energy[s] += op.energy
        sat_time[s]   += op.duration

    # Sólo para medir, no reoptimiza: evalúa/penaliza si algo quedó mal
    return x

def run_baseline(n_sat=5, n_targets=60, n_ops=800, seed=0, inst=None):
    """
    Ejecuta el baseline Greedy y devuelve instancia, solución y métricas.

    Args:
        n_sat (int): Satélites si se genera instancia.
        n_targets (int): Objetivos si se genera instancia.
        n_ops (int): Oportunidades si se genera instancia.
        seed (int): Semilla de aleatoriedad.
        inst (Instance|None): Instancia opcional para reutilizar.

    Returns:
        tuple: (instancia, vector solución, objetivos, meta con ratios/energía/makespan).
    """
    if inst is None:
        inst = gen_instance(n_sat, n_targets, n_ops, seed)
    x = greedy_solution(inst, seed)
    objs = evaluate_vector(x, inst)
    _, _, meta = repair_and_score(x, inst, return_meta=True)
    return inst, x, objs, meta
