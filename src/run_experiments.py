import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from pymoo.performance_indicator.hv import Hypervolume as _HV
    def make_hv(ref_point):
        return _HV(ref_point=ref_point)
except ImportError:  # fallback for older pymoo versions
    from pymoo.indicators.hv import HV as _HV
    class make_hv:
        def __init__(self, ref_point):
            self._hv = _HV(ref_point=ref_point)
        def do(self, F):
            if hasattr(self._hv, "do"):
                return self._hv.do(F)
            return self._hv(F)

from .data import gen_instance
from .greedy import run_baseline
from .nsga2_run import run_nsga2

ENERGY_MAX = 200.0
TIME_MAX = 600.0
HV_REF = np.array([1.0, 1.0, 1.0])
SUMMARY_WEIGHTS = np.array([0.5, 0.35, 0.15])

def ensure_out():
    """Crea el directorio de salida `outputs` si no existe."""
    os.makedirs("outputs", exist_ok=True)

def normalize_front(F, inst):
    """
    Normaliza objetivos para cálculo de hipervolumen y retorna ratio de cobertura.

    Args:
        F (array-like): Objetivos reparados (neg_coverage, energy, makespan).
        inst (Instance): Instancia con max_expected_coverage para escalar cobertura.

    Returns:
        tuple[np.ndarray, np.ndarray]: Matriz normalizada (cov_obj, energy_obj, time_obj)
        y vector coverage_ratio en [0,1].
    """
    arr = np.atleast_2d(np.array(F, dtype=float))
    max_cov = max(inst.max_expected_coverage, 1e-9)
    coverage_ratio = np.clip((-arr[:,0]) / max_cov, 0.0, 1.0)
    cov_obj = 1.0 - coverage_ratio
    energy_obj = np.clip(arr[:,1] / ENERGY_MAX, 0.0, 1.0)
    time_obj = np.clip(arr[:,2] / TIME_MAX, 0.0, 1.0)
    norm = np.column_stack((cov_obj, energy_obj, time_obj))
    return norm, coverage_ratio

def pick_solution_cov_energy(cov_ratio, F_rep, tol=0.90):
    """
    Elige una solución NSGA-II representativa privilegiando cobertura y baja energía.

    Args:
        cov_ratio (array-like): Coberturas relativas reparadas (0-1) para cada solución.
        F_rep (array-like): Objetivos reparados (neg_coverage, energy, makespan).
        tol (float): Umbral de cobertura mínima respecto al máximo (por defecto 0.90).

    Returns:
        int: Índice de la solución seleccionada.
    """
    cov = np.array(cov_ratio)
    max_cov = np.max(cov)
    eligible = np.where(cov >= tol * max_cov)[0]
    if eligible.size == 0:
        eligible = np.where(cov == max_cov)[0]
    energies = F_rep[eligible, 1]
    best_idx = eligible[int(np.argmin(energies))]
    return int(best_idx)


def pick_solution_vs_greedy_3way(cov_ratio, F_rep, greedy_cov, greedy_energy, greedy_makespan,
                                 makespan_slack=1.00): 
    """
    Elige una solución que mejore Greedy en energía y makespan; si no, prioriza makespan.

    Args:
        cov_ratio (array-like): Coberturas relativas reparadas (0-1).
        F_rep (array-like): Objetivos reparados.
        greedy_cov (float): Cobertura de Greedy (0-1).
        greedy_energy (float): Energía de Greedy.
        greedy_makespan (float): Makespan de Greedy.
        makespan_slack (float): Factor de holgura para aceptar makespan (1.0 = igual).

    Returns:
        int: Índice de la solución elegida.
    """
    cov = np.array(cov_ratio)
    E = np.array(F_rep[:, 1])
    M = np.array(F_rep[:, 2])

    eligible = np.where((E <= greedy_energy) & (M <= greedy_makespan * makespan_slack))[0]
    if eligible.size:
        cov_sub = cov[eligible]
        best = eligible[int(np.argmax(cov_sub))]
        return int(best)
    # fallback: prioriza makespan bajo
    eligible = np.where(E <= greedy_energy)[0]
    if eligible.size:
        M_sub = M[eligible]
        best = eligible[int(np.argmin(M_sub))]
        return int(best)
    # último recurso: tu política anterior
    return pick_solution_energy_cap(cov_ratio, F_rep, energy_cap=greedy_energy, fallback_tol=0.95)


def pick_solution_energy_cap(cov_ratio, F_rep, energy_cap, fallback_tol=0.90):
    """
    Elige máxima cobertura bajo un tope de energía; desempata por menor energía.

    Args:
        cov_ratio (array-like): Coberturas relativas (0-1).
        F_rep (array-like): Objetivos reparados.
        energy_cap (float): Tope de energía permitido.
        fallback_tol (float): Tolerancia de cobertura para el fallback.

    Returns:
        int: Índice seleccionado.
    """
    cov = np.array(cov_ratio)
    energy = np.array(F_rep[:, 1])
    eligible = np.where(energy <= energy_cap)[0]
    if eligible.size:
        cov_sub = cov[eligible]
        max_cov = np.max(cov_sub)
        top = eligible[np.where(cov_sub == max_cov)[0]]
        return int(top[int(np.argmin(energy[top]))])
    return pick_solution_cov_energy(cov_ratio, F_rep, tol=fallback_tol)


def pick_solution_balanced(cov_ratio, F_rep, w_cov=0.6, w_energy=0.4):
    """
    Elige una solución equilibrando cobertura y energía sin cap duro.

    Args:
        cov_ratio (array-like): Coberturas relativas (0-1).
        F_rep (array-like): Objetivos reparados.
        w_cov (float): Peso de la cobertura (por defecto 0.6).
        w_energy (float): Peso de la energía (por defecto 0.4).

    Returns:
        int: Índice de la solución con menor score ponderado.
    """
    cov_ratio = np.array(cov_ratio)
    energies = np.array(F_rep[:, 1])
    cov_obj = 1.0 - cov_ratio
    energy_obj = energies / max(ENERGY_MAX, 1e-9)
    scores = w_cov * cov_obj + w_energy * energy_obj
    return int(np.argmin(scores))

def plot_front(nsga_df, greedy_df):
    """
    Dibuja y guarda la gráfica de cobertura vs energía para NSGA-II y Greedy.

    Args:
        nsga_df (pd.DataFrame): Frentes de NSGA-II con columnas coverage_pct y energy.
        greedy_df (pd.DataFrame): Resultados de Greedy con coverage_pct y energy.
    """
    plt.figure()
    if not nsga_df.empty:
        plt.scatter(nsga_df["coverage_pct"], nsga_df["energy"], s=12, label="NSGA-II")
    if not greedy_df.empty:
        plt.scatter(greedy_df["coverage_pct"], greedy_df["energy"], s=60, marker="^", label="Greedy")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Energy (units)")
    plt.title("Cobertura vs Energía (Greedy vs NSGA-II)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/pareto_greedy_nsga.png")
    plt.close()

def main():
    """Ejecuta la campaña experimental: Greedy vs NSGA-II, genera CSVs y gráficas."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat", type=int, default=5)
    ap.add_argument("--targets", type=int, default=60)
    ap.add_argument("--ops", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--pop_size", type=int, default=200, help="NSGA-II population size")
    ap.add_argument("--n_gen", type=int, default=300, help="NSGA-II generations")
    args = ap.parse_args()

    ensure_out()

    greedy_rows = []
    nsga_front_rows = []
    nsga_summary = []
    hv_runs = []
    norm_all = []

    for seed in range(args.seeds):
        inst = gen_instance(args.sat, args.targets, args.ops, seed=seed)

        _, _, fg, meta_g = run_baseline(args.sat, args.targets, args.ops, seed=seed, inst=inst)
        greedy_rows.append({
            "seed": seed,
            "coverage_pct": meta_g["coverage_ratio"] * 100.0,
            "energy": fg[1],
            "makespan": fg[2]
        })

        inst_nsga, _, _, F_rep = run_nsga2(
            args.sat,
            args.targets,
            args.ops,
            seed=seed,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            inst=inst,
        )
        norm_front, cov_ratio = normalize_front(F_rep, inst_nsga)
        hv = make_hv(HV_REF).do(norm_front)
        hv_runs.append(hv)
        norm_all.append(norm_front)

        for idx, objs in enumerate(F_rep):
            nsga_front_rows.append({
                "seed": seed,
                "solution": idx,
                "coverage_pct": cov_ratio[idx] * 100.0,
                "energy": objs[1],
                "makespan": objs[2]
            })

        # Representativo: máxima cobertura dentro de un cap energético sobre Greedy (10% arriba)
        energy_cap = fg[1] * 1.10
        chosen_idx = pick_solution_energy_cap(cov_ratio, F_rep, energy_cap=energy_cap, fallback_tol=0.90)



        nsga_summary.append({
            "seed": seed,
            "coverage_pct": cov_ratio[chosen_idx] * 100.0,
            "energy": F_rep[chosen_idx,1],
            "makespan": F_rep[chosen_idx,2],
            "hv": hv
        })

    df_greedy = pd.DataFrame(greedy_rows)
    df_nsga_front = pd.DataFrame(nsga_front_rows)
    df_nsga = pd.DataFrame(nsga_summary)

    df_greedy.to_csv("outputs/baseline.csv", index=False)
    df_nsga_front.to_csv("outputs/front_nsga2.csv", index=False)

    plot_front(df_nsga_front, df_greedy)

    summary_records = []
    for name, df in [("Greedy", df_greedy), ("NSGA-II", df_nsga)]:
        if df.empty:
            continue
        rec = {
            "algorithm": name,
            "coverage_pct": df["coverage_pct"].mean(),
            "energy": df["energy"].mean(),
            "makespan": df["makespan"].mean()
        }
        if "hv" in df.columns:
            rec["hypervolume"] = df["hv"].mean()
        summary_records.append(rec)
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv("outputs/summary_metrics.csv", index=False)

    hv_global = float("nan")
    if norm_all:
        hv_global = make_hv(HV_REF).do(np.vstack(norm_all))

    success_metrics = {
        "nsga_mean_hv": float(np.mean(hv_runs)) if hv_runs else None,
        "nsga_global_hv": hv_global,
        "coverage_gain_vs_greedy": float(df_nsga["coverage_pct"].mean() - df_greedy["coverage_pct"].mean()) if (not df_nsga.empty and not df_greedy.empty) else None,
        "energy_delta_vs_greedy": float(df_nsga["energy"].mean() - df_greedy["energy"].mean()) if (not df_nsga.empty and not df_greedy.empty) else None
    }
    with open("outputs/success_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(success_metrics, fh, indent=2)

if __name__ == "__main__":
    main()
