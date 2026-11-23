import numpy as np
from .common import normalize_weights

def optimize(objective, dimensions=3, wolves=30, iterations=100, seed=42):
    rng = np.random.default_rng(seed)
    population = np.array([normalize_weights(rng.random(dimensions))
                           for _ in range(wolves)])
    history = []

    for t in range(iterations):
        scores = np.array([objective(ind) for ind in population])
        order = np.argsort(scores)
        alpha, beta, delta = (population[order[0]].copy(),
                              population[order[1]].copy(),
                              population[order[2]].copy())
        best_score = float(scores[order[0]])
        history.append(best_score)
        a = 2 - 2 * (t / max(iterations - 1, 1))
        updated = []
        for wolf in population:
            candidates = []
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(dimensions), rng.random(dimensions)
                A = 2 * a * r1 - a
                C = 2 * r2
                candidates.append(leader - A * np.abs(C * leader - wolf))
            updated.append(normalize_weights(np.mean(candidates, axis=0)))
        population = np.array(updated)

    scores = np.array([objective(ind) for ind in population])
    idx = int(np.argmin(scores))
    return population[idx], float(scores[idx]), history
