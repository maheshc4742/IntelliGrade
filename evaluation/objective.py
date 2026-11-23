import numpy as np
from .scoring import predict_marks

def mae_objective(frame):
    def objective(weights):
        predictions = predict_marks(frame, weights)
        actual = frame["human_marks"].to_numpy(dtype=float)
        return float(np.mean(np.abs(actual - predictions)))
    return objective
