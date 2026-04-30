"""
B2B Distribution Problem — Greedy Approximation Solver

Selects facilities one at a time using a cost-effectiveness ratio:

    score(f) = |new clients covered by f| / cost(f)

This is the standard greedy for the Weighted Set Cover problem and achieves
an approximation ratio of  H(n) = 1 + 1/2 + ... + 1/n ≈ ln(n) + 1
relative to the optimal solution cost, where n is the number of clients.

Time complexity : O(m · n) per round → O(m² · n) overall
                  m = # facilities, n = # clients

The budget B is enforced as a hard constraint: a facility is only selected
if it does not push the running total over B.  If all clients cannot be
covered within budget the solver reports a partial solution with a warning.
"""

import math
import random
import time


# ── Helpers ──────────────────────────────────────────────────────────────────

def euclidean(p1, p2):
    """Euclidean distance between two (x, y) points (km)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ── Instance Generation (shared with brute-force.py) ─────────────────────────

def generate_instance(n_clients, m_facilities, radius=150, seed=None):
    """
    Generate a random B2B facility-location instance.

    Returns
    -------
    clients    : list of (x, y)            — client site coordinates (km)
    facilities : list of (x, y, cost)      — candidate site coords + annual cost ($)
    coverage   : dict {facility_idx: set of client indices within radius}
    """
    rng = random.Random(seed)

    clients = [
        (rng.uniform(0, 500), rng.uniform(0, 500))
        for _ in range(n_clients)
    ]

    facilities = [
        (rng.uniform(0, 500), rng.uniform(0, 500), round(rng.uniform(50_000, 500_000), 2))
        for _ in range(m_facilities)
    ]

    coverage = {}
    for fi, (fx, fy, _) in enumerate(facilities):
        coverage[fi] = {
            ci for ci, (cx, cy) in enumerate(clients)
            if euclidean((fx, fy), (cx, cy)) <= radius
        }

    return clients, facilities, coverage


# ── Greedy Solver ─────────────────────────────────────────────────────────────

def greedy(clients, facilities, coverage, budget):
    """
    Greedy cost-effectiveness approximation for the B2B facility location problem.

    Algorithm
    ---------
    1. Maintain a set of *uncovered* clients (initially all clients).
    2. At each round compute, for every remaining facility f:
           score(f) = |coverage(f) ∩ uncovered| / cost(f)
       This is the number of *new* clients covered per dollar spent.
    3. Select the facility with the highest score that fits within the
       remaining budget.
    4. Remove covered clients from the uncovered set; repeat until all
       clients are covered or no budget-feasible facility remains.

    Returns
    -------
    selected   : list of chosen facility indices
    total_cost : float — sum of selected facility costs
    covered    : int   — number of clients actually covered
    feasible   : bool  — True iff all clients are covered within budget
    """
    n          = len(clients)
    uncovered  = set(range(n))
    selected   = []
    total_cost = 0.0

    # Working copy: facility_idx → set of clients it can still newly cover
    remaining = {fi: set(cov) for fi, cov in coverage.items()}

    while uncovered:
        # Find the best budget-feasible facility by cost-effectiveness score.
        best_fi    = None
        best_score = -1.0

        for fi, cov in remaining.items():
            cost = facilities[fi][2]
            if total_cost + cost > budget:
                continue                         # would exceed budget — skip
            new_coverage = len(cov & uncovered)
            if new_coverage == 0:
                continue                         # covers nothing new — skip
            score = new_coverage / cost
            if score > best_score:
                best_score = score
                best_fi    = fi

        if best_fi is None:
            # No facility can improve coverage within the remaining budget.
            break

        # Commit to this facility.
        selected.append(best_fi)
        total_cost += facilities[best_fi][2]
        uncovered  -= remaining[best_fi]
        del remaining[best_fi]

    feasible = len(uncovered) == 0
    covered  = n - len(uncovered)

    return selected, round(total_cost, 2), covered, feasible


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("B2B Distribution Problem — Greedy Approximation Solver")
    print("=" * 60)

    # Small configs: compare greedy against brute-force quality
    SMALL_CONFIGS = [
        (5,  8,  0.5),
        (6,  10, 0.6),
        (8,  12, 0.5),
        (10, 14, 0.5),
        (10, 16, 0.4),
    ]

    # Large configs: demonstrate scalability (brute-force cannot reach these)
    LARGE_CONFIGS = [
        (50,   100,  0.4),
        (100,  200,  0.4),
        (500,  1000, 0.4),
        (1000, 2000, 0.4),
        (2000, 4000, 0.4),
    ]

    print("\n── Small instances (comparable to brute-force) ──")
    print(f"{'n':>5} {'m':>5} {'budget':>14} │ {'k':>4} {'cost':>14} {'covers':>8} {'feasible':>9} {'time':>9}")
    print("─" * 75)
    for nc, mf, bm in SMALL_CONFIGS:
        clients, facilities, coverage = generate_instance(nc, mf, radius=150, seed=42)
        budget = round(sum(f[2] for f in facilities) * bm, 2)

        t0 = time.perf_counter()
        subset, cost, covered, feasible = greedy(clients, facilities, coverage, budget)
        elapsed = time.perf_counter() - t0

        print(
            f"{nc:>5} {mf:>5} ${budget:>13,.0f} │ "
            f"{len(subset):>4} ${cost:>13,.0f} {covered:>4}/{nc:<3} "
            f"{'YES' if feasible else 'NO ':>9} {elapsed:.5f}s"
        )

    print("\n── Large instances (greedy only — brute-force infeasible) ──")
    print(f"{'n':>5} {'m':>6} {'budget':>14} │ {'k':>5} {'cost':>14} {'covers':>9} {'feasible':>9} {'time':>10}")
    print("─" * 80)
    for nc, mf, bm in LARGE_CONFIGS:
        clients, facilities, coverage = generate_instance(nc, mf, radius=150, seed=42)
        budget = round(sum(f[2] for f in facilities) * bm, 2)

        t0 = time.perf_counter()
        subset, cost, covered, feasible = greedy(clients, facilities, coverage, budget)
        elapsed = time.perf_counter() - t0

        print(
            f"{nc:>5} {mf:>6} ${budget:>13,.0f} │ "
            f"{len(subset):>5} ${cost:>13,.0f} {covered:>5}/{nc:<4} "
            f"{'YES' if feasible else 'NO ':>9} {elapsed:.5f}s"
        )
