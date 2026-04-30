"""
B2B Distribution Problem — Brute-Force Exact Solver

Exhaustively enumerates every subset of candidate facilities (2^m total) to
find the minimum-cost feasible subset that:
  1. Covers all client sites within the reach radius R.
  2. Does not exceed the operational budget B.

Time complexity : O(2^m · n)   m = # facilities, n = # clients
Space complexity: O(m + n)

This is the "exact" / certificate-verifiable solution.  It is only practical
for small instances (m ≤ ~20) because the search space grows exponentially.
"""

import itertools
import math
import random
import time


# ── Helpers ──────────────────────────────────────────────────────────────────

def euclidean(p1, p2):
    """Euclidean distance between two (x, y) points (km)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ── Instance Generation ──────────────────────────────────────────────────────

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


# ── Certificate Checker ───────────────────────────────────────────────────────

def verify(subset, n_clients, facilities, coverage, budget):
    """
    Polynomial-time certificate verifier.

    Returns True iff the given subset:
      - Covers every client site, AND
      - Has a total operational cost ≤ budget.
    """
    all_clients = set(range(n_clients))
    covered = set().union(*(coverage[fi] for fi in subset))
    total_cost = sum(facilities[fi][2] for fi in subset)
    return all_clients <= covered and total_cost <= budget


# ── Brute-Force Solver ────────────────────────────────────────────────────────

def brute_force(clients, facilities, coverage, budget):
    """
    Enumerate all 2^m subsets of candidate facilities.

    Strategy
    --------
    Iterate subset sizes from 1 → m.  Within each size, iterate all C(m, size)
    combinations.  Track the feasible subset with minimum total cost.

    Early-exit optimisation: once a feasible subset at size k is found, no
    subset of size > k can have a strictly lower cost than the best so far
    (because adding more facilities only increases cost), so we break after
    size k is fully explored.

    Returns
    -------
    best_subset : list of facility indices  (or None if no solution exists)
    best_cost   : total operational cost of best_subset  (or None)
    stats       : dict with diagnostic counters
    """
    n = len(clients)
    m = len(facilities)
    all_clients = set(range(n))

    best_subset = None
    best_cost   = float("inf")
    subsets_evaluated = 0

    for size in range(1, m + 1):
        found_at_size = False

        for subset in itertools.combinations(range(m), size):
            subsets_evaluated += 1

            total_cost = sum(facilities[fi][2] for fi in subset)
            if total_cost >= best_cost:          # prune: already worse than best
                continue
            if total_cost > budget:              # prune: over budget
                continue

            covered = set()
            for fi in subset:
                covered |= coverage[fi]

            if all_clients <= covered:
                best_subset = list(subset)
                best_cost   = total_cost
                found_at_size = True

        # Once we've explored all subsets of this size and have a winner,
        # no larger subset can improve the cost (costs are positive), so stop.
        if found_at_size:
            break

    stats = {
        "subsets_evaluated": subsets_evaluated,
        "worst_case_2m":     2 ** m,
    }
    return best_subset, (round(best_cost, 2) if best_subset else None), stats


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("B2B Distribution Problem — Brute-Force Exact Solver")
    print("=" * 60)

    CONFIGS = [
        # (n_clients, m_facilities, budget_multiplier)
        (5,  8,  0.5),
        (6,  10, 0.6),
        (8,  12, 0.5),
        (10, 14, 0.5),
        (10, 16, 0.4),
    ]

    for nc, mf, bm in CONFIGS:
        clients, facilities, coverage = generate_instance(nc, mf, radius=150, seed=42)

        # Budget = bm × sum of all facility costs (forces real selection trade-offs)
        total_possible_cost = sum(f[2] for f in facilities)
        budget = round(total_possible_cost * bm, 2)

        t0 = time.perf_counter()
        subset, cost, stats = brute_force(clients, facilities, coverage, budget)
        elapsed = time.perf_counter() - t0

        if subset is not None:
            ok = verify(subset, nc, facilities, coverage, budget)
            coverage_count = len(set().union(*(coverage[fi] for fi in subset)))
            print(
                f"n={nc:3d}  m={mf:3d}  budget=${budget:>12,.0f} | "
                f"k={len(subset)}  cost=${cost:>12,.0f}  "
                f"covers={coverage_count}/{nc}  "
                f"verified={'YES' if ok else 'NO'}  "
                f"subsets={stats['subsets_evaluated']:,}/{stats['worst_case_2m']:,}  "
                f"t={elapsed:.4f}s"
            )
        else:
            print(
                f"n={nc:3d}  m={mf:3d}  budget=${budget:>12,.0f} | "
                f"NO FEASIBLE SOLUTION  t={elapsed:.4f}s"
            )
