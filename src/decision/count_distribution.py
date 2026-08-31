"""Dynamic programming helpers for Poisson-binomial count distributions."""

import numpy as np


def prefix_count_distributions(probabilities):
    """Return distributions for positive counts in every leading prefix.

    The element at index ``length`` is a vector of length ``length + 1`` whose
    entry ``count`` is the probability of exactly ``count`` positives among
    the first ``length`` conditionally independent Bernoulli labels.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")
    distributions = [np.array([1.0], dtype=np.float64)]
    for probability in values:
        previous = distributions[-1]
        current = np.zeros(previous.size + 1, dtype=np.float64)
        current[:-1] += previous * (1.0 - probability)
        current[1:] += previous * probability
        distributions.append(current)
    return tuple(distributions)


def suffix_count_distributions(probabilities):
    """Return positive-count distributions for every trailing suffix."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")
    n_labels = values.size
    distributions = [None] * (n_labels + 1)
    distributions[n_labels] = np.array([1.0], dtype=np.float64)
    for start in range(n_labels - 1, -1, -1):
        following = distributions[start + 1]
        current = np.zeros(following.size + 1, dtype=np.float64)
        current[:-1] += following * (1.0 - values[start])
        current[1:] += following * values[start]
        distributions[start] = current
    return tuple(distributions)
