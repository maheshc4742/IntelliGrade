import numpy as np
from .scoring import predict_marks

def calculate_metrics(frame, weights):
    actual = frame["human_marks"].to_numpy(dtype=float)
    predicted = predict_marks(frame, weights)
    errors = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "Exact match (%)": float(np.mean(np.isclose(predicted, actual)) * 100),
        "Within 0.5 (%)": float(np.mean(np.abs(errors) <= 0.5) * 100),
    }
