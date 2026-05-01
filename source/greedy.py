"""
B2B Facility Location Problem — Greedy Approximation

Selects facilities one at a time using a cost-effectiveness ratio:

    score(f) = |new clients covered by f| / cost(f)

This is the standard greedy solution for the Weighted Set Cover problem and achieves an approximation ratio of  H(n) = 1 + 1/2 + ... + 1/n ≈ ln(n) + 1 relative to the optimal solution cost, where n is the number of clients.

Time Complexity : O(m · n) per round → O(m² · n) overall
                  m = # facilities, n = # clients

The budget B is enforced as a hard constraint: a facility is only selected if it does not push the running total over B.  If all clients cannot be covered within budget the solver reports a partial solution with a warning.

Strategy 
--------
1. Maintain a set of uncovered clients (initially all clients).
2. At each round compute, for every remaining facility f: score(f) = |coverage(f) ∩ uncovered| / cost(f) This is the number of *new* clients covered per dollar spent.
3. Select the facility with the highest score that does not exceed the remaining budget.
4. Update the uncovered set and remaining budget. Repeat until all clients are covered or no more facilities can be selected within budget.


Args:
    clients    : list of (x, y)            — client site coordinates (km)
    facilities : list of (x, y, cost)      — candidate site coords + annual cost ($)
    coverage   : dict {facility_idx: set of client indices within radius}
    budget     : float                    — maximum total cost allowed ($)

Returns:
    selected_facilities : list of chosen facility indices
    total_cost          : float — sum of selected facility costs
    covered_clients     : set of client indices covered by the selected facilities
    feasible            : bool — True if all clients are covered within budget, False otherwise
"""

import math
import random
import time

def greedy(clients, facilities, coverage, budget):
    n_clients      = len(clients)
    uncovered      = set(range(n_clients))
    selected_facilities = []
    total_cost     = 0.0

    # Working copy: clients still needing coverage, and facilities not yet selected.
    remaining = {fi: set(cov) for fi, cov in coverage.items()}

    while uncovered:
        best_fi    = None
        best_score = -1.0

        for fi, cov in remaining.items():
            cost = facilities[fi][2]
            new_coverage = len(cov & uncovered)

            if total_cost + cost > budget or new_coverage == 0:
                continue                         # would exceed budget or covers nothing new — skip

            score = new_coverage / cost
            if score > best_score:
                best_score = score
                best_fi    = fi
            
        if best_fi is None:
            # No facility can improve coverage within the remaining budget.
            break

        # Commit to this facility.
        selected_facilities.append(best_fi)
        total_cost += facilities[best_fi][2]
        uncovered -= remaining[best_fi]
        del remaining[best_fi]

    feasible = len(uncovered) == 0
    covered_clients = set(range(n_clients)) - uncovered

    return selected_facilities, round(total_cost, 2), covered_clients, feasible

