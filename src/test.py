"""
B2B Supply Chain – Facility Location Experiments
Exact (brute-force) vs Greedy approximation
"""

import itertools, time, random, math, json, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

random.seed(42)

# ─────────────────────────────────────────────────
# 1. Instance generator
# ─────────────────────────────────────────────────
def generate_instance(n_clients, m_facilities, coverage_radius=150, seed=None):
    """
    Returns (clients, facilities, coverage) where:
      clients     = list of (lat, lon)
      facilities  = list of (lat, lon)
      coverage    = dict  facility_idx -> set of client_idx it covers
    Geography: random points in a 500x500 km bounding box
    """
    rng = random.Random(seed)
    clients    = [(rng.uniform(0, 500), rng.uniform(0, 500)) for _ in range(n_clients)]
    facilities = [(rng.uniform(0, 500), rng.uniform(0, 500)) for _ in range(m_facilities)]
    coverage   = {}
    for fi, (fx, fy) in enumerate(facilities):
        covered = set()
        for ci, (cx, cy) in enumerate(clients):
            dist = math.hypot(fx - cx, fy - cy)
            if dist <= coverage_radius:
                covered.add(ci)
        coverage[fi] = covered
    return clients, facilities, coverage

def is_feasible(instance):
    """Check that every client is covered by at least one facility."""
    _, _, coverage = instance
    all_clients = set(range(len(_[0] if isinstance(_[0], tuple) else _)))
    union = set().union(*coverage.values())
    return all_clients <= union

# ─────────────────────────────────────────────────
# 2. Exact solver — brute-force enumeration
# ─────────────────────────────────────────────────
def exact_solver(n_clients, coverage):
    """
    Enumerate all 2^m subsets of facilities.
    Return (best_subset, best_size).
    """
    m = len(coverage)
    all_clients = set(range(n_clients))
    best = None

    for size in range(1, m + 1):
        if best is not None:
            break
        for subset in itertools.combinations(range(m), size):
            covered = set()
            for fi in subset:
                covered |= coverage[fi]
            if all_clients <= covered:
                best = list(subset)
                break

    return best, len(best) if best else m

# ─────────────────────────────────────────────────
# 3. Greedy approximation solver
# ─────────────────────────────────────────────────
def greedy_solver(n_clients, coverage):
    """
    At each step pick the facility covering the most uncovered clients.
    Approximation ratio: ln(n) + 1
    """
    uncovered = set(range(n_clients))
    selected  = []
    remaining = dict(coverage)   # fi -> set of all clients it covers

    while uncovered:
        # pick facility with max coverage of still-uncovered clients
        best_fi   = max(remaining, key=lambda fi: len(remaining[fi] & uncovered))
        best_gain = len(remaining[best_fi] & uncovered)
        if best_gain == 0:
            break   # infeasible instance
        selected.append(best_fi)
        uncovered -= remaining[best_fi]
        del remaining[best_fi]

    return selected, len(selected)

# ─────────────────────────────────────────────────
# 4. Benchmarking
# ─────────────────────────────────────────────────
EXACT_CONFIGS = [
    # (n_clients, m_facilities)
    (5,  5),
    (6,  6),
    (7,  7),
    (8,  8),
    (8,  10),
    (10, 12),
    (10, 14),
    (10, 16),
    (12, 18),
    (12, 20),
]

GREEDY_CONFIGS = [
    (10,  20),
    (20,  40),
    (50,  100),
    (100, 200),
    (200, 400),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
]

def bench_exact():
    rows = []
    for nc, mf in EXACT_CONFIGS:
        clients, facilities, coverage = generate_instance(nc, mf, seed=7)
        t0 = time.perf_counter()
        subset, opt_k = exact_solver(nc, coverage)
        elapsed = time.perf_counter() - t0
        subsets_checked = sum(math.comb(mf, k) for k in range(1, opt_k + 1))
        rows.append({
            "n_clients": nc,
            "m_facilities": mf,
            "opt_facilities": opt_k,
            "time_s": round(elapsed, 6),
            "subsets_worst": 2**mf,
        })
        print(f"Exact  n={nc:3d} m={mf:3d}  k*={opt_k}  t={elapsed:.4f}s  2^m={2**mf:,}")
    return rows

def bench_greedy():
    rows = []
    for nc, mf in GREEDY_CONFIGS:
        clients, facilities, coverage = generate_instance(nc, mf, seed=7)
        t0 = time.perf_counter()
        subset, gr_k = greedy_solver(nc, coverage)
        elapsed = time.perf_counter() - t0
        rows.append({
            "n_clients": nc,
            "m_facilities": mf,
            "greedy_facilities": gr_k,
            "time_s": round(elapsed, 6),
        })
        print(f"Greedy n={nc:4d} m={mf:4d}  k_greedy={gr_k}  t={elapsed:.6f}s")
    return rows

def bench_comparison():
    """Run both on same instances for quality comparison."""
    configs = EXACT_CONFIGS[:8]
    rows = []
    for nc, mf in configs:
        clients, facilities, coverage = generate_instance(nc, mf, seed=7)

        t0 = time.perf_counter()
        _, opt_k = exact_solver(nc, coverage)
        t_exact = time.perf_counter() - t0

        t0 = time.perf_counter()
        _, gr_k = greedy_solver(nc, coverage)
        t_greedy = time.perf_counter() - t0

        ratio = gr_k / opt_k if opt_k > 0 else 1.0
        rows.append({
            "n_clients": nc,
            "m_facilities": mf,
            "opt_k": opt_k,
            "greedy_k": gr_k,
            "approx_ratio": round(ratio, 3),
            "t_exact_s": round(t_exact, 6),
            "t_greedy_s": round(t_greedy, 6),
            "speedup": round(t_exact / t_greedy, 1) if t_greedy > 0 else 999,
        })
    return rows

# ─────────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────────
COLORS = {"exact": "#1F5C99", "greedy": "#E8792A", "ratio": "#27AE60", "subsets": "#8E44AD"}

def plot_exact_runtime(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')

    m_vals = [r["m_facilities"] for r in rows]
    t_vals = [r["time_s"] for r in rows]
    s_vals = [r["subsets_worst"] for r in rows]

    ax1.plot(m_vals, t_vals, "o-", color=COLORS["exact"], lw=2, ms=7)
    ax1.fill_between(m_vals, t_vals, alpha=0.12, color=COLORS["exact"])
    ax1.set_xlabel("Number of candidate facilities (m)", fontsize=12)
    ax1.set_ylabel("Runtime (seconds)", fontsize=12)
    ax1.set_title("Exact Algorithm — Runtime vs m", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(m_vals, s_vals, "s--", color=COLORS["subsets"], lw=2, ms=7)
    ax2.set_xlabel("Number of candidate facilities (m)", fontsize=12)
    ax2.set_ylabel("Worst-case subsets (2ᵐ) — log scale", fontsize=12)
    ax2.set_title("Search Space Growth (Exponential)", fontsize=13, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    plt.savefig("plot_exact.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_exact.png")

def plot_greedy_runtime(rows):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')

    nc_vals = [r["n_clients"] for r in rows]
    t_vals  = [r["time_s"] for r in rows]

    ax.plot(nc_vals, t_vals, "o-", color=COLORS["greedy"], lw=2, ms=7)
    ax.fill_between(nc_vals, t_vals, alpha=0.12, color=COLORS["greedy"])
    ax.set_xlabel("Number of clients (n)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Greedy Algorithm — Scales to Large Inputs", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_greedy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_greedy.png")

def plot_comparison(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')

    labels = [f"n={r['n_clients']}\nm={r['m_facilities']}" for r in rows]
    x = range(len(rows))

    # --- Solution quality ---
    opt_k = [r["opt_k"] for r in rows]
    gr_k  = [r["greedy_k"] for r in rows]
    w = 0.35
    bars1 = ax1.bar([xi - w/2 for xi in x], opt_k,  width=w, label="Optimal (exact)",  color=COLORS["exact"],  alpha=0.85)
    bars2 = ax1.bar([xi + w/2 for xi in x], gr_k,   width=w, label="Greedy approx.",   color=COLORS["greedy"], alpha=0.85)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Facilities opened (k)", fontsize=11)
    ax1.set_title("Solution Quality: Optimal vs Greedy", fontsize=12, fontweight="bold")
    ax1.legend(); ax1.grid(True, axis="y", alpha=0.3)

    # --- Runtime comparison ---
    t_exact  = [r["t_exact_s"]*1000 for r in rows]
    t_greedy = [r["t_greedy_s"]*1000 for r in rows]
    ax2.bar([xi - w/2 for xi in x], t_exact,  width=w, label="Exact (ms)",  color=COLORS["exact"],  alpha=0.85)
    ax2.bar([xi + w/2 for xi in x], t_greedy, width=w, label="Greedy (ms)", color=COLORS["greedy"], alpha=0.85)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Runtime (milliseconds)", fontsize=11)
    ax2.set_title("Runtime: Exact vs Greedy (same instances)", fontsize=12, fontweight="bold")
    ax2.legend(); ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_comparison.png")

# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Exact benchmarks ===")
    exact_rows = bench_exact()

    print("\n=== Greedy benchmarks ===")
    greedy_rows = bench_greedy()

    print("\n=== Comparison ===")
    comp_rows = bench_comparison()

    # Save JSON for doc generation
    with open("results.json", "w") as f:
        json.dump({"exact": exact_rows, "greedy": greedy_rows, "comparison": comp_rows}, f, indent=2)

    plot_exact_runtime(exact_rows)
    plot_greedy_runtime(greedy_rows)
    plot_comparison(comp_rows)

    print("\nAll done.")