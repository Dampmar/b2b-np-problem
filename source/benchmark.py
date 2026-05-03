"""
B2B Facility Location Problem - Paired Benchmark Suite

Compares the Greedy approximation against the Brute-force exact solver on the
same randomly generated instances (same seed, clients, facilities, radius,
budget) across a sweep of facility counts m.

For every (m, trial) pair we record:
    - greedy_time           wall-clock seconds for the greedy solver
    - bf_time               wall-clock seconds for the brute force solver
    - greedy_cost           total cost of greedy's selection (if feasible)
    - bf_cost               total cost of the exact optimum (if one exists)
    - greedy_feasible       True iff greedy covered every client within budget
    - bf_feasible           True iff an exact feasible solution exists
    - cost_ratio            greedy_cost / bf_cost when both are feasible

These per-trial values are then aggregated per m to build a summary table
similar to the "Greedy vs Brute Force on the same instances" analysis.

Usage (from the source/ folder):
    python benchmark.py
"""

import importlib.util
import os
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np

# Set Global Font Family for all plots
plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.size"] = 12
plt.rcParams["font.serif"] = ['Georgia']

from helpers import generate_instance
from greedy import greedy

# brute-force.py has a hyphen so it can't be imported with a normal import statement.
_bf_path = os.path.join(os.path.dirname(__file__), "brute-force.py")
_spec = importlib.util.spec_from_file_location("brute_force_module", _bf_path)
_bf_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bf_module)
brute_force = _bf_module.brute_force


# ── Experiment configuration ──────────────────────────────────────────────────

N_CLIENTS    = 12       # fixed across the sweep
M_RANGE      = range(8, 19)   # m = 8, 9, ..., 18  (inclusive)
RADIUS       = 200
BUDGET_FRAC  = 0.5      # budget = BUDGET_FRAC * sum(facility costs)
TRIALS_PER_M = 5        # independent instances per m value
BASE_SEED    = 2026     # deterministic seeding → reproducible results


def run_single_trial(n_clients, m_facilities, radius, budget_frac, seed):
    """Generate one instance and solve it with both algorithms."""
    clients, facilities, coverage = generate_instance(
        n_clients, m_facilities, radius=radius, seed=seed
    )
    budget = round(sum(f[2] for f in facilities) * budget_frac, 2)

    # Greedy
    t0 = time.perf_counter()
    g_selected, g_cost, g_covered, g_feasible = greedy(
        clients, facilities, coverage, budget
    )
    g_time = time.perf_counter() - t0

    # Brute force (exact)
    t0 = time.perf_counter()
    bf_selected, bf_cost, bf_stats = brute_force(
        clients, facilities, coverage, budget
    )
    bf_time = time.perf_counter() - t0
    bf_feasible = bf_selected is not None

    return {
        "m":              m_facilities,
        "seed":           seed,
        "budget":         budget,
        "greedy_time":    g_time,
        "greedy_cost":    g_cost if g_feasible else None,
        "greedy_feasible": g_feasible,
        "bf_time":        bf_time,
        "bf_cost":        bf_cost if bf_feasible else None,
        "bf_feasible":    bf_feasible,
        "bf_subsets":     bf_stats["subsets_evaluated"],
    }


def aggregate(trials):
    """Collapse a list of per-trial dicts into summary stats for one m."""
    g_times  = [t["greedy_time"] for t in trials]
    bf_times = [t["bf_time"]     for t in trials]

    # Greedy is "feasible" here only when brute force also had a feasible
    # optimum to compare against; otherwise the ratio is undefined.
    both_feasible = [
        t for t in trials
        if t["greedy_feasible"] and t["bf_feasible"] and t["bf_cost"] > 0
    ]
    cost_ratios = [t["greedy_cost"] / t["bf_cost"] for t in both_feasible]

    g_feasible_rate = sum(1 for t in trials if t["greedy_feasible"]) / len(trials)

    return {
        "m":                  trials[0]["m"],
        "trials":             len(trials),
        "greedy_avg_time":    statistics.fmean(g_times),
        "bf_avg_time":        statistics.fmean(bf_times),
        "speedup":            (statistics.fmean(bf_times) /
                               statistics.fmean(g_times)) if g_times else 0,
        "greedy_feasible_rate": g_feasible_rate,
        "avg_cost_ratio":     statistics.fmean(cost_ratios) if cost_ratios else None,
        "min_cost_ratio":     min(cost_ratios) if cost_ratios else None,
        "max_cost_ratio":     max(cost_ratios) if cost_ratios else None,
        "paired_feasible":    len(both_feasible),
    }


def plot_bf_scatter(all_rows):
    """Scatter plot of m vs brute-force avg time with exponential trendline."""
    ms = np.array([r["m"] for r in all_rows])
    bf_times = np.array([r["bf_avg_time"] for r in all_rows])
    gd_times = np.array([r["greedy_avg_time"] for r in all_rows])

    # Fit exponential: t = a * exp(b * m)  →  ln(t) = ln(a) + b*m
    log_times = np.log(bf_times)
    b, ln_a = np.polyfit(ms, log_times, 1)
    a = np.exp(ln_a)

    m_fit = np.linspace(ms.min(), ms.max(), 300)
    t_fit = a * np.exp(b * m_fit)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ms, bf_times, color="r", zorder=3, label="Observed (avg over trials)")      
    ax.plot(m_fit, t_fit, color="pink", linewidth=2, label=rf"Best Fit Trendline: $t = {a:.2e} \cdot e^{{{b:.3f}\,m}}$")       
    ax.scatter(ms, gd_times, color="b", zorder=3, label="Greedy (avg over trials)")        
    ax.set_xlabel("Number of facilities $m$", fontsize=12)
    ax.set_ylabel("Brute-force avg time (s)", fontsize=12)
    ax.set_title("Brute-force runtime $s$ vs. Number of Facilities $m$", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("bf_runtime_scatter.png", dpi=300)
    plt.show()


def main():
    print("=" * 78)
    print("B2B Facility Location — Paired Benchmark (Greedy vs Brute Force)")
    print("=" * 78)
    print(f"  Clients per instance : {N_CLIENTS}")
    print(f"  Facility sweep (m)   : {M_RANGE.start}..{M_RANGE.stop - 1}")
    print(f"  Trials per m         : {TRIALS_PER_M}")
    print(f"  Radius               : {RADIUS} km")
    print(f"  Budget fraction      : {BUDGET_FRAC:.2f} of total facility cost")
    print(f"  Base seed            : {BASE_SEED}")
    print()

    all_rows = []
    raw_trials = []

    for m in M_RANGE:
        m_trials = []
        for k in range(TRIALS_PER_M):
            seed = BASE_SEED + 1000 * m + k   # unique per (m, trial)
            trial = run_single_trial(
                N_CLIENTS, m, RADIUS, BUDGET_FRAC, seed
            )
            m_trials.append(trial)
            raw_trials.append(trial)

        summary = aggregate(m_trials)
        all_rows.append(summary)

        print(
            f"  m={m:>2}  "
            f"greedy~{summary['greedy_avg_time']*1e6:8.1f}us  "
            f"bf~{summary['bf_avg_time']*1e3:9.3f}ms  "
            f"speedup={summary['speedup']:8.1f}x  "
            f"g_feas={summary['greedy_feasible_rate']:.2f}  "
            + (f"cost_ratio={summary['avg_cost_ratio']:.4f}"
               if summary['avg_cost_ratio'] is not None else "cost_ratio=  n/a")
        )

    # ── Final table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Paired benchmark results")
    print("=" * 78)
    header = (
        f"{'m':>3} | {'Greedy avg (s)':>14} | {'BF avg (s)':>12} | "
        f"{'Speedup':>9} | {'G feas.':>7} | {'G/Opt cost':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in all_rows:
        ratio_str = (f"{r['avg_cost_ratio']:.4f}"
                     if r['avg_cost_ratio'] is not None else "   n/a ")
        print(
            f"{r['m']:>3} | {r['greedy_avg_time']:>14.6f} | "
            f"{r['bf_avg_time']:>12.6f} | "
            f"{r['speedup']:>8.1f}x | "
            f"{r['greedy_feasible_rate']:>7.2f} | "
            f"{ratio_str:>11}"
        )

    # ── Top-level takeaways ───────────────────────────────────────────────────
    speedups = [r['speedup'] for r in all_rows]
    ratios_all = [r['avg_cost_ratio'] for r in all_rows
                  if r['avg_cost_ratio'] is not None]

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"  Compared sizes             : m = {M_RANGE.start}..{M_RANGE.stop - 1}")
    print(f"  Max speedup (Greedy vs BF) : {max(speedups):,.1f}x  (at m={all_rows[speedups.index(max(speedups))]['m']})")
    print(f"  Min speedup                : {min(speedups):,.1f}x  (at m={all_rows[speedups.index(min(speedups))]['m']})")
    if ratios_all:
        print(f"  Greedy/Optimal cost ratio  : {min(ratios_all):.4f} .. {max(ratios_all):.4f}")
        print(f"  Overall avg cost ratio     : {statistics.fmean(ratios_all):.4f}")
        # Greedy is always ≥ optimum, so the "overhead" is ratio - 1.
        over = [r - 1 for r in ratios_all]
        print(f"  Greedy overhead vs optimum : {min(over)*100:.2f}% .. {max(over)*100:.2f}%")

    # Number of trials where greedy matched the exact optimum.
    exact_matches = sum(
        1 for t in raw_trials
        if (t["greedy_feasible"] and t["bf_feasible"]
            and abs(t["greedy_cost"] - t["bf_cost"]) < 1e-6)
    )
    paired = sum(1 for t in raw_trials if t["greedy_feasible"] and t["bf_feasible"])
    if paired:
        print(f"  Greedy matched optimum     : {exact_matches}/{paired} paired trials "
              f"({100*exact_matches/paired:.1f}%)")

    plot_bf_scatter(all_rows)


if __name__ == "__main__":
    main()
