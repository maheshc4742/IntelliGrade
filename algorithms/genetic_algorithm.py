import numpy as np
from .common import normalize_weights

def optimize(objective, dimensions=3, population_size=40, generations=100, seed=42):
    rng = np.random.default_rng(seed)
    population = np.array([normalize_weights(rng.random(dimensions))
                           for _ in range(population_size)])
    history = []

    def tournament(pop, scores, k=3):
        ids = rng.integers(0, len(pop), size=k)
        return pop[ids[np.argmin(scores[ids])]].copy()

    for _ in range(generations):
        scores = np.array([objective(ind) for ind in population])
        history.append(float(scores.min()))
        elite_ids = np.argsort(scores)[:2]
        new_population = [population[i].copy() for i in elite_ids]
        while len(new_population) < population_size:
            p1, p2 = tournament(population, scores), tournament(population, scores)
            alpha = rng.random()
            child = alpha * p1 + (1 - alpha) * p2
            if rng.random() < 0.35:
                child += rng.normal(0, 0.12, dimensions)
            new_population.append(normalize_weights(child))
        population = np.array(new_population)

    final_scores = np.array([objective(ind) for ind in population])
    best = int(np.argmin(final_scores))
    return population[best], float(final_scores[best]), history
