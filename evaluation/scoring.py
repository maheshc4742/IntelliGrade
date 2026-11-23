import numpy as np

FEATURES = ["semantic_similarity", "rubric_coverage", "keypoint_coverage"]

def predict_marks(frame, weights):
    x = frame[FEATURES].to_numpy(dtype=float)
    max_marks = frame["max_marks"].to_numpy(dtype=float)
    raw = (x @ np.asarray(weights, dtype=float)) * max_marks
    # Preserve the project's half-mark quantization rule.
    return np.clip(np.round(raw * 2) / 2, 0, max_marks)
