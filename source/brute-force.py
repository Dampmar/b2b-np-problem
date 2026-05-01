
"""
B2B Facility Location Problem — Brute Force Exact Solver
    
Strategy
--------
Iterate subset sizes from 1 -> m. Within each size, iterate all C(m, size) subsets of facilities. For each subset:
  - Check if it covers all clients and is within budget.
  - If valid, compute its cost and update the best solution if it's cheaper.

Pruning: skip any subset whose total cost already meets or exceeds the current best, or exceeds the budget.

Args:
    clients    : list of (x, y)            — client site coordinates (km)
    facilities : list of (x, y, cost)      — candidate site coords + annual cost ($)
    coverage   : dict {facility_idx: set of client indices within radius}
    budget     : float                    — maximum total cost allowed ($)

Returns:
    best_subset : list of chosen facility indices
    best_cost   : float — sum of selected facility costs
    stats       : dict — diagnostic counters (e.g., subsets evaluated)
"""

import itertools
import math 
import random 
import time 

def brute_force(clients, facilities, coverage, budget):
    n = len(clients)
    m = len(facilities)
    all_clients = set(range(n))

    best_subset = None
    best_cost   = float("inf")
    subsets_evaluated = 0
    
    for size in range(1, m + 1):

        # Iterate all subsets of this size.
        for subset in itertools.combinations(range(m), size):
            subsets_evaluated += 1

            total_cost = sum(facilities[fi][2] for fi in subset)

            # Early exit pruning: if cost already exceeds best or budget, skip.
            if total_cost >= best_cost or total_cost > budget:
                continue

            covered = set()
            for fi in subset:
                covered |= coverage[fi]

            if all_clients <= covered:
                best_subset = list(subset)
                best_cost   = total_cost
    
    stats = {
        "subsets_evaluated": subsets_evaluated,
        "worst_case_2m":     2 ** m,
    }
    return best_subset, (round(best_cost, 2) if best_subset else None), stats