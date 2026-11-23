import numpy as np

def normalize_weights(x):
    """Project a vector onto the non-negative unit simplex."""
    x = np.maximum(np.asarray(x, dtype=float), 1e-12)
    total = x.sum()
    return x / total if total else np.ones_like(x) / len(x)

def evaluate_population(population, objective):
    return np.asarray([objective(individual) for individual in population], dtype=float)
