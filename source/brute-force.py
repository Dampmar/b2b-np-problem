import itertools
import math 
import random 
import time 

def brute_force(clients, facilities, coverage, budget):
    """
    Brute-force solver for the B2B facility location problem.
    
    Strategy
    --------
    Iterate subset sizes from 1 -> m. Within each size, iterate all C(m, size) subsets of facilities. For each subset:
      - Check if it covers all clients and is within budget.
      - If valid, compute its cost and update the best solution if it's cheaper.

    Early-exit optimisation: once a feasible subset at size k is found, no subset of size > k can have a strictly lower cost than the best so far (because adding more facilities only increases cost), so we break after size k is fully explored.

    Returns
    -------
    best_subset : list of chosen facility indices
    best_cost   : float — sum of selected facility costs
    dict        : diagnostic counters (e.g., subsets evaluated, time taken)
    """
    n = len(clients)
    m = len(facilities)
    all_clients = set(range(n))

    best_subset = None
    best_cost   = float("inf")
    subsets_evaluated = 0
    
    start = time.time()
    for size in range(1, m + 1):
        found_at_size = False 

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
                best_subset   = list(subset)
                best_cost     = total_cost
                found_at_size = True

        # Once we've explored all subsets of this size and have a winner, no need to check larger subsets. Because costs are positive, adding more facilities can only increase cost, so larger subsets can't be better than the best found at this size.
        if found_at_size:
            break
    end = time.time()
    
    stats = {
        "subsets_evaluated": subsets_evaluated,
        "worst_case_2m":     2 ** m,
        "time_taken_sec":   round(end - start, 2)
    }
    return best_subset, (round(best_cost, 2) if best_subset else None), stats