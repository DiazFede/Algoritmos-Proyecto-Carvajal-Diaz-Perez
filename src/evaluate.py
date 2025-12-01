
import numpy as np
from collections import defaultdict
from .data import Instance

def expected_value(op, targets):
    """
    Calcula el valor esperado de una oportunidad.

    Args:
        op (Opportunity): Oportunidad de observación.
        targets (list): Lista de objetivos.

    Returns:
        float: Valor = prioridad * cov.
    """
    return targets[op.tid].priority * op.cov


def repair_and_score(x, inst: Instance, return_meta: bool = False):
    """
    Repara un vector binario y calcula objetivos (neg_coverage, energía, makespan).

    - Elimina solapes por satélite y excesos de energía/tiempo.
    - Penaliza falta de cobertura en objetivos críticos (aumenta makespan).

    Args:
        x (array-like): Vector binario de selección.
        inst (Instance): Instancia con satélites, objetivos y oportunidades.
        return_meta (bool): Si True, devuelve meta con coberturas/penalizaciones.

    Returns:
        tuple: (vector reparado, objetivos) o (vector reparado, objetivos, meta) si return_meta=True.
    """
    x = np.array(x, dtype=int)
    ops = inst.ops

    # Group opportunities by satellite
    by_sat = {s: [] for s in range(len(inst.sats))}
    for i, sel in enumerate(x):
        if sel:
            by_sat[ops[i].sat].append(ops[i])
    for s in by_sat:
        by_sat[s].sort(key=lambda o: o.start)

    repaired = np.zeros_like(x)
    energy_total = 0.0
    neg_coverage = 0.0  # we'll subtract coverage to minimize
    coverage_total = 0.0
    makespan = 0.0

    # track per-satellite
    last_theta = {}

    targets = inst.targets
    first_finish = {}
    target_obs_count = defaultdict(int)
    observed_targets = set()

    # enforce per-satellite feasibility greedily by value density
# enforce per-satellite feasibility WITHOUT re-optimizing:
# keep the order already in by_sat (we sorted by start above)
    for s, ops_s in by_sat.items():

        ops_s_sorted = sorted(
            ops_s,
            key=lambda o: (
                expected_value(o, targets) / max(o.energy, 1e-6),
                1.0 if targets[o.tid].critical else 0.0,   # favorece críticos
                -o.end                                      # y los que terminan antes
            ),
            reverse=True,
        )

        schedule = []
        e_used = 0.0
        t_used = 0.0
        e_budget = inst.sats[s].e_max
        t_budget = inst.sats[s].t_max
        slew_coef = inst.sats[s].slew_coeff

        timeline = []
        for cand in ops_s_sorted:
            # check overlap with accepted timeline
            feasible = True
            for (st, en) in timeline:
                if not (cand.end <= st or cand.start >= en):
                    feasible = False
                    break
            if not feasible:
                continue
            # energy/time limits (approx)
            if e_used + cand.energy > e_budget or t_used + cand.duration > t_budget:
                continue
            # accept
            schedule.append(cand)
            timeline.append((cand.start, cand.end))
            e_used += cand.energy
            t_used += cand.duration

        # contribute to totals
        schedule.sort(key=lambda o:o.start)
        prev_theta = last_theta.get(s, None)
        for idx, o in enumerate(schedule):
            repaired[o.oid] = 1
            val = expected_value(o, targets)
            neg_coverage -= val
            coverage_total += val
            target_obs_count[o.tid] += 1
            observed_targets.add(o.tid)

            # sin costos de slew / theta
            energy_total += o.energy

            if targets[o.tid].critical and (o.tid not in first_finish or o.end < first_finish[o.tid]):
                first_finish[o.tid] = o.end


            # makespan over critical targets
            if targets[o.tid].critical and (o.tid not in first_finish or o.end < first_finish[o.tid]):
                first_finish[o.tid] = o.end

        # después de construir 'schedule' y antes de sumar al total:
        if schedule:
            # colapsa múltiples observaciones por target crítico a la más temprana
            earliest_by_crit_tid = {}
            for o in schedule:
                if targets[o.tid].critical:
                    if (o.tid not in earliest_by_crit_tid) or (o.end < earliest_by_crit_tid[o.tid].end):
                        earliest_by_crit_tid[o.tid] = o
            # marca todas como aceptadas, pero si querés ser más agresivo,
            # podés descartar las observaciones críticas posteriores del mismo tid.


    if first_finish:
        makespan = max(first_finish.values())
    else:
        makespan = 0.0

    # enforce coverage of critical targets (constraint Σ x_o ≥ m_t)
    horizon = getattr(inst, "horizon", 600.0)
    crit_shortfall = 0
    for tg in targets:
        required = getattr(tg, "min_obs", 1 if getattr(tg, "critical", False) else 0)
        observed = target_obs_count.get(tg.tid, 0)
        if observed < required:
            miss = required - observed
            crit_shortfall += miss
            makespan += (horizon + 10.0) * miss  # penalize infeasible plans

    max_cov = max(getattr(inst, "max_expected_coverage", 1.0), 1e-9)
    coverage_ratio = coverage_total / max_cov

    if return_meta:
        meta = {
            "coverage": coverage_total,
            "coverage_ratio": coverage_ratio,
            "energy": energy_total,
            "makespan": makespan,
            "crit_shortfall": crit_shortfall,
            "unique_targets": len(observed_targets),
            "total_targets": len(targets),
            "crit_targets": sum(1 for t in targets if getattr(t, "critical", False)),
        }
        return repaired, (neg_coverage, energy_total, makespan), meta

    return repaired, (neg_coverage, energy_total, makespan)

def evaluate_vector(x, inst: Instance):
    """
    Evalúa un vector binario reparándolo y retornando los objetivos.

    Args:
        x (array-like): Selección binaria de oportunidades.
        inst (Instance): Instancia de planificación.

    Returns:
        tuple: Objetivos (neg_coverage, energy_total, makespan_crit).
    """
    _, objs = repair_and_score(x, inst)
    return objs
