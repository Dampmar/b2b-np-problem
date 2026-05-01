import importlib.util
import os
import time

from helpers import generate_instance, print_solution
from greedy import greedy

# brute-force.py has a hyphen so it can't be imported with a normal import statement.
_bf_path = os.path.join(os.path.dirname(__file__), "brute-force.py")
_spec = importlib.util.spec_from_file_location("brute_force_module", _bf_path)
_bf_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bf_module)
brute_force = _bf_module.brute_force


# ── Trial configuration ───────────────────────────────────────────────────────

N_CLIENTS   = 12
M_FACILITIES = 25
RADIUS      = 200
BUDGET_FRAC = 0.5   # fraction of total facility cost used as budget
SEED        = None

# ── Generate instance ─────────────────────────────────────────────────────────

clients, facilities, coverage = generate_instance(
    N_CLIENTS, M_FACILITIES, radius=RADIUS, seed=SEED
)
budget = round(sum(f[2] for f in facilities) * BUDGET_FRAC, 2)

print("=" * 60)
print("B2B Facility Location — Trial Run")
print("=" * 60)
print(f"  Clients    : {N_CLIENTS}")
print(f"  Facilities : {M_FACILITIES}")
print(f"  Radius     : {RADIUS} km")
print(f"  Budget     : ${budget:,.2f}  ({BUDGET_FRAC*100:.0f}% of total facility cost)")

# ── Greedy solver ─────────────────────────────────────────────────────────────

print("\n── Greedy ──────────────────────────────────────────────")
t0 = time.perf_counter()
g_selected, g_cost, g_covered, g_feasible = greedy(clients, facilities, coverage, budget)
g_elapsed = time.perf_counter() - t0

print_solution(g_selected, g_cost, g_covered, g_feasible)
print(f"Time: {g_elapsed:.5f}s")

# ── Brute-force solver ────────────────────────────────────────────────────────

print("\n── Brute Force ─────────────────────────────────────────")
t0 = time.perf_counter()
bf_selected, bf_cost, bf_stats = brute_force(clients, facilities, coverage, budget)
bf_elapsed = time.perf_counter() - t0

if bf_selected is not None:
    bf_covered = set()
    for fi in bf_selected:
        bf_covered |= coverage[fi]
    bf_feasible = len(bf_covered) == N_CLIENTS
    print_solution(bf_selected, bf_cost, bf_covered, bf_feasible)
else:
    print("\n=== Solution Summary ===")
    print("No feasible solution found within budget.")

print(f"Subsets evaluated : {bf_stats['subsets_evaluated']:,} / {bf_stats['worst_case_2m']:,} worst-case")
print(f"Time: {bf_elapsed:.5f}s")
