#functions taken and slightly modified from FunSearch, a project by DeepMind under the Apache License 2.0
import numpy as np
# list the functions that can be imported from helpers.py
__all__ = [
    'l1_bound',
    'first_fit_heuristic',
    'best_fit_heuristic',
    'discovered_heuristic_or',
    'discovered_heuristic_weibull',
    'funsearch_heuristic_weibull'
]


# --- L1 lower bound functions ---
def l1_bound(items, capacity):
    return np.ceil(np.sum(items) / capacity)

def l1_bound_dataset(instances):
    l1_bounds = []
    for name in instances:
        instance = instances[name]
        l1_bounds.append(l1_bound(instance['items'], instance['capacity']))
    return np.mean(l1_bounds)

# --- Heuristics ---
def first_fit_heuristic(item, bins):
    scores = np.zeros(len(bins))
    if len(scores) > 0:
        scores[0] = 1
    return scores

def best_fit_heuristic(item, bins):
    return -(np.array(bins) - item)

def discovered_heuristic_or(item, bins):

    def s(bin, item):
        if bin - item <= 2:
            return 4
        elif (bin - item) <= 3:
            return 3
        elif (bin - item) <= 5:
            return 2
        elif (bin - item) <= 7:
            return 1
        elif (bin - item) <= 9:
            return 0.9
        elif (bin - item) <= 12:
            return 0.95
        elif (bin - item) <= 15:
            return 0.97
        elif (bin - item) <= 18:
            return 0.98
        elif (bin - item) <= 20:
            return 0.98
        elif (bin - item) <= 21:
            return 0.98
        else:
            return 0.99
    return np.array([s(b, item) for b in bins])


def discovered_heuristic_weibull(item, bins):
    bins = np.array(bins)
    max_bin_cap = max(bins)
    score = (bins - max_bin_cap)**2 / item + bins**2 / (item**2)
    score += bins**2 / item**3
    score[bins > item] = -score[bins > item]
    score[1:] -= score[:-1]
    return score

def funsearch_heuristic_weibull(item: float, bins: np.ndarray) -> np.ndarray:
    """
    The first heuristic we implemented, which shows strong general performance and
    was noted for its results on Weibull datasets in the paper's analysis.
    """
    score = 1000 * np.ones(bins.shape)
    score -= bins * (bins - item)
    index = np.argmin(bins)
    score[index] *= item
    score[index] -= (bins[index] - item)**4
    return score

# --- Verification helper ---
def is_valid_packing(packing, items, capacity):
    packed_items = sum(packing, [])
    if sorted(packed_items) != sorted(items):
        return False
    for bin_items in packing:
        if sum(bin_items) > capacity:
            return False
    return True


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]