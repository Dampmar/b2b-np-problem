import math
import random

"""
B2B Facility Location Problem - Helper Functions 

Euclidean:
    Function to compute the Euclidean distance between two points (x, y).

Instance Generation:
    Function to generate random instances of the B2B facility location problem, including:
    - Random client locations within a 500x500 km area.
    - Random facility locations and costs within the same area.
    - Precomputed coverage sets for each facility based on a specified radius.

 

"""

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def generate_instance(n_clients, m_facilities, radius=150, seed=None):
    rng = random.Random(seed)

    # Randomize client generation - locations 500x500 km area.
    clients = [
        (rng.uniform(0, 500), rng.uniform(0, 500))
        for _ in range(n_clients)
    ]

    # Randomize facility generation - locations 500x500 km area + costs.
    facilities = [
        (rng.uniform(0, 500), rng.uniform(0, 500), round(rng.uniform(50_000, 500_000), 2))
        for _ in range(m_facilities)
    ]

    # Precompute coverage: which clients are within radius of each facility.
    coverage = {}
    for fi, (fx, fy, _) in enumerate(facilities):
        coverage[fi] = {
            ci for ci, (cx, cy) in enumerate(clients)
            if euclidean((fx, fy), (cx, cy)) <= radius
        }
    
    return clients, facilities, coverage

def print_solution(selected_facilities, total_cost, covered_clients, feasible):

    status_msg = "Feasible solution found." if feasible \
        else "Partial solution (not all clients covered), consider incrementing budget."

    print("\n=== Solution Summary ===")
    print(status_msg)
    print(f"Selected facilities: {selected_facilities}")
    print(f"Total cost: ${total_cost:,.2f}")
    print(f"Covered clients: {len(covered_clients)} (out of total)")