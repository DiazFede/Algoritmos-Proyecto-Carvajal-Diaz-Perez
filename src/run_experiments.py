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
from .mosa import run_mosa
from .tabu import tabu_improve

ENERGY_MAX = 100
TIME_MAX = 600.0
HV_REF = np.array([1.0, 1.0, 1.0])
SUMMARY_WEIGHTS = np.array([0.5, 0.35, 0.15])

def ensure_out():
    os.makedirs("outputs", exist_ok=True)

def normalize_front(F, inst):
    """Return normalized objectives for HV + coverage ratio for reporting."""
    arr = np.atleast_2d(np.array(F, dtype=float))
    max_cov = max(inst.max_expected_coverage, 1e-9)
    coverage_ratio = np.clip((-arr[:,0]) / max_cov, 0.0, 1.0)
    cov_obj = 1.0 - coverage_ratio
    energy_obj = np.clip(arr[:,1] / ENERGY_MAX, 0.0, 1.0)
    time_obj = np.clip(arr[:,2] / TIME_MAX, 0.0, 1.0)
    norm = np.column_stack((cov_obj, energy_obj, time_obj))
    return norm, coverage_ratio

def pick_solution(norm):
    scores = norm.dot(SUMMARY_WEIGHTS)
    return int(np.argmin(scores))

def plot_front(df, fname, title):
    if df.empty:
        return
    plt.figure()
    plt.scatter(df["coverage_pct"], df["energy"], s=14)
    plt.xlabel("Coverage (%)")
    plt.ylabel("Energy (units)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_all_fronts(nsga_df, mosa_df, greedy_df, tabu_df):
    plt.figure()
    if not nsga_df.empty:
        plt.scatter(nsga_df["coverage_pct"], nsga_df["energy"], s=12, label="NSGA-II")
    if not mosa_df.empty:
        plt.scatter(mosa_df["coverage_pct"], mosa_df["energy"], s=12, marker="x", label="MOSA")
    if not greedy_df.empty:
        plt.scatter(greedy_df["coverage_pct"], greedy_df["energy"], s=60, marker="^", label="Greedy")
    if not tabu_df.empty:
        plt.scatter(tabu_df["coverage_pct"], tabu_df["energy"], s=60, marker="s", label="NSGA-II + Tabu")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Energy (units)")
    plt.title("Coverage vs Energy (all methods)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/pareto_all.png")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat", type=int, default=5)
    ap.add_argument("--targets", type=int, default=60)
    ap.add_argument("--ops", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()

    ensure_out()

    greedy_rows = []
    nsga_front_rows = []
    nsga_summary = []
    mosa_front_rows = []
    mosa_summary = []
    tabu_rows = []
    hv_runs = []
    norm_all = []

    for seed in range(args.seeds):
        inst = gen_instance(args.sat, args.targets, args.ops, seed=seed)

        _, xg, fg, meta_g = run_baseline(args.sat, args.targets, args.ops, seed=seed, inst=inst)
        greedy_rows.append({
            "seed": seed,
            "coverage_pct": meta_g["coverage_ratio"] * 100.0,
            "energy": fg[1],
            "makespan": fg[2]
        })

        inst_nsga, res, Xr, F_rep = run_nsga2(args.sat, args.targets, args.ops, seed=seed, inst=inst)
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

        chosen_idx = pick_solution(norm_front)
        nsga_summary.append({
            "seed": seed,
            "coverage_pct": cov_ratio[chosen_idx] * 100.0,
            "energy": F_rep[chosen_idx,1],
            "makespan": F_rep[chosen_idx,2],
            "hv": hv
        })

        # Tabu applied on best energy plan above 90% coverage (fallback: best coverage)
        eligible = np.where(cov_ratio >= 0.90)[0]
        if eligible.size == 0:
            eligible = np.array([int(np.argmax(cov_ratio))])
        cand_idx = eligible[np.argmin(F_rep[eligible,1])]
        xt, ft = tabu_improve(inst_nsga, Xr[cand_idx], seed=seed + 1000)
        norm_tabu, cov_ratio_tabu = normalize_front(ft, inst_nsga)
        tabu_rows.append({
            "seed": seed,
            "coverage_pct": cov_ratio_tabu[0] * 100.0,
            "energy": ft[1],
            "makespan": ft[2]
        })

        sols_mosa, F_mosa = run_mosa(inst, seed=seed, iters=3000)
        if len(F_mosa):
            norm_mosa, cov_ratio_mosa = normalize_front(F_mosa, inst)
            for idx, objs in enumerate(F_mosa):
                mosa_front_rows.append({
                    "seed": seed,
                    "solution": idx,
                    "coverage_pct": cov_ratio_mosa[idx] * 100.0,
                    "energy": objs[1],
                    "makespan": objs[2]
                })
            mosa_idx = pick_solution(norm_mosa)
            mosa_summary.append({
                "seed": seed,
                "coverage_pct": cov_ratio_mosa[mosa_idx] * 100.0,
                "energy": F_mosa[mosa_idx,1],
                "makespan": F_mosa[mosa_idx,2]
            })

    df_greedy = pd.DataFrame(greedy_rows)
    df_nsga_front = pd.DataFrame(nsga_front_rows)
    df_nsga = pd.DataFrame(nsga_summary)
    df_mosa_front = pd.DataFrame(mosa_front_rows)
    df_mosa = pd.DataFrame(mosa_summary)
    df_tabu = pd.DataFrame(tabu_rows)

    df_greedy.to_csv("outputs/baseline.csv", index=False)
    df_nsga_front.to_csv("outputs/front_nsga2.csv", index=False)
    df_mosa_front.to_csv("outputs/mosa.csv", index=False)
    df_tabu.to_csv("outputs/tabu.csv", index=False)

    plot_front(df_nsga_front, "outputs/pareto_nsga2.png", "NSGA-II Pareto (coverage vs energy)")
    plot_front(df_mosa_front, "outputs/pareto_mosa.png", "MOSA Pareto (coverage vs energy)")
    plot_all_fronts(df_nsga_front, df_mosa_front, df_greedy, df_tabu)

    summary_records = []
    for name, df in [("Greedy", df_greedy), ("MOSA", df_mosa), ("NSGA-II", df_nsga), ("NSGA-II + Tabu", df_tabu)]:
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
        "tabu_coverage_success_rate": float((df_tabu["coverage_pct"] >= 90.0).mean()) if not df_tabu.empty else None,
        "energy_reduction_vs_greedy": float(df_greedy["energy"].mean() - df_tabu["energy"].mean()) if (not df_greedy.empty and not df_tabu.empty) else None
    }
    with open("outputs/success_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(success_metrics, fh, indent=2)

if __name__ == "__main__":
    main()
