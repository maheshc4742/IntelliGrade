import numpy as np
from .common import normalize_weights

def optimize(objective, dimensions=3, particles=30, iterations=100, seed=42):
    rng = np.random.default_rng(seed)
    positions = rng.random((particles, dimensions))
    positions = np.array([normalize_weights(p) for p in positions])
    velocities = rng.normal(0, 0.08, size=(particles, dimensions))

    personal_best = positions.copy()
    personal_scores = np.array([objective(p) for p in positions])
    best_idx = int(np.argmin(personal_scores))
    global_best = personal_best[best_idx].copy()
    global_score = float(personal_scores[best_idx])
    history = [global_score]

    inertia, cognitive, social = 0.65, 1.5, 1.5
    for _ in range(iterations):
        r1 = rng.random((particles, dimensions))
        r2 = rng.random((particles, dimensions))
        velocities = (inertia * velocities
                      + cognitive * r1 * (personal_best - positions)
                      + social * r2 * (global_best - positions))
        positions = np.array([normalize_weights(p + v)
                              for p, v in zip(positions, velocities)])
        scores = np.array([objective(p) for p in positions])
        improved = scores < personal_scores
        personal_best[improved] = positions[improved]
        personal_scores[improved] = scores[improved]
        idx = int(np.argmin(personal_scores))
        if personal_scores[idx] < global_score:
            global_score = float(personal_scores[idx])
            global_best = personal_best[idx].copy()
        history.append(global_score)
    return global_best, global_score, history
