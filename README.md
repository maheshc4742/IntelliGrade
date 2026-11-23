# IntelliGrade
## Bio-Inspired Optimization of AI-Based Answer Evaluation

IntelliGrade is a **standalone educational BIA project** that compares Particle Swarm Optimization (PSO), Genetic Algorithm (GA), and Grey Wolf Optimizer (GWO) for tuning answer-evaluation weights.

It does not depend on ScriptSense, CodeBERT, Gemini, OCR, or any external API.

### What it optimizes
Each answer row contains three normalized evaluation signals:
- `semantic_similarity`: semantic alignment between student and reference answers (0–1)
- `rubric_coverage`: proportion of rubric criteria satisfied (0–1)
- `keypoint_coverage`: proportion of expected key points covered (0–1)

The optimizer searches for non-negative weights that sum to 1. A weighted score is converted to marks and quantized to 0.5 increments. The objective is to minimize Mean Absolute Error (MAE) against teacher-assigned marks.

**Important:** the included dataset is synthetic demonstration data, not real student assessment data. Replace it with teacher-verified data before making research claims.

### Project structure
```text
IntelliGrade/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_answers.csv
├── algorithms/
│   ├── __init__.py
│   ├── common.py
│   ├── pso.py
│   ├── genetic_algorithm.py
│   └── gwo.py
├── evaluation/
│   ├── __init__.py
│   ├── scoring.py
│   ├── objective.py
│   └── metrics.py
├── experiments/
│   ├── __init__.py
│   └── run_comparison.py
└── outputs/
```

### Setup (Windows)
```powershell
cd IntelliGrade
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### Dataset format
CSV columns:
`semantic_similarity,rubric_coverage,keypoint_coverage,human_marks,max_marks`

All three feature columns must be in [0, 1]. Human marks must be between 0 and max marks.

### Experiment methodology
1. Split records into training, validation, and test sets (70/15/15).
2. Optimize weights using training data only.
3. Select the best candidate on validation data.
4. Report final metrics on the held-out test set.
5. Compare the three optimizers with an equal-weight baseline.

The demonstration app uses a fixed random seed for reproducibility. For a formal study, repeat each optimizer across multiple random seeds and report mean and standard deviation.

### Metrics
- MAE: mean absolute error in marks
- RMSE: root mean squared error in marks
- Exact-match accuracy: proportion of predictions exactly matching teacher marks
- Within-0.5 accuracy: proportion within half a mark

The implementation rounds predictions to the nearest 0.5 mark. In an actual grading deployment, explicit numeric rubric checkpoints should remain deterministic and should not be overridden by optimized weights.

### Limitations
- Synthetic sample data only.
- The feature scores are assumed to have been computed already; this project does not perform OCR or semantic embedding extraction.
- The optimizers tune a small set of weights; they do not train a language model.
- Metrics are meaningful only when evaluated against reliable human-assigned marks.
