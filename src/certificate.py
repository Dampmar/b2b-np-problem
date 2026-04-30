def is_feasible(instance):
    """ Checks that all clients are covered by at least one facility. """
    _, _, coverage = instance
    all_clients = set(range(len(_[0] if isinstance(_[0], tuple) else _)))
    union = set().union(*coverage.values())
    return all_clients <= union