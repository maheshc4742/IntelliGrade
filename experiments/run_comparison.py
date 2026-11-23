import pandas as pd
from sklearn.model_selection import train_test_split
from algorithms.pso import optimize as run_pso
from algorithms.genetic_algorithm import optimize as run_ga
from algorithms.gwo import optimize as run_gwo
from evaluation.objective import mae_objective
from evaluation.metrics import calculate_metrics

def run_experiment(data, seed=42, iterations=100):
    train_val, test = train_test_split(data, test_size=0.15, random_state=seed)
    train, validation = train_test_split(train_val, test_size=(0.15 / 0.85),
                                         random_state=seed)
    objective = mae_objective(train)
    optimizers = {
        "PSO": lambda: run_pso(objective, iterations=iterations, seed=seed),
        "GA": lambda: run_ga(objective, generations=iterations, seed=seed),
        "GWO": lambda: run_gwo(objective, iterations=iterations, seed=seed),
    }
    candidates = {"Equal-weight baseline": [1/3, 1/3, 1/3]}
    histories = {}
    for name, fn in optimizers.items():
        weights, _, history = fn()
        candidates[name] = weights
        histories[name] = history

    # Choose the best optimizer on validation MAE; test set is untouched until reporting.
    val_scores = {
        name: calculate_metrics(validation, weights)["MAE"]
        for name, weights in candidates.items()
    }
    chosen = min((n for n in candidates if n != "Equal-weight baseline"),
                 key=lambda n: val_scores[n])
    rows = []
    for name, weights in candidates.items():
        row = {"Algorithm": name, **calculate_metrics(test, weights)}
        row.update({f"Weight: {feature}": float(weight)
                    for feature, weight in zip(
                        ["Semantic", "Rubric", "Key points"], weights)})
        rows.append(row)
    return pd.DataFrame(rows), candidates, histories, chosen, val_scores
