import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.run_comparison import run_experiment

st.set_page_config(page_title="IntelliGrade", page_icon="🧬", layout="wide")
st.title("🧬 IntelliGrade")
st.subheader("Bio-Inspired Optimization of AI-Based Answer Evaluation")
st.caption("Standalone BIA project • PSO vs Genetic Algorithm vs Grey Wolf Optimizer")

st.info("The bundled CSV is synthetic demonstration data. Use teacher-verified records for genuine research evaluation.")

default_path = Path(__file__).parent / "data" / "sample_answers.csv"
uploaded = st.file_uploader("Upload evaluation dataset (CSV)", type=["csv"])
data = pd.read_csv(uploaded) if uploaded else pd.read_csv(default_path)

required = ["semantic_similarity", "rubric_coverage", "keypoint_coverage", "human_marks", "max_marks"]
missing = [c for c in required if c not in data.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()
if data.empty:
    st.error("Dataset is empty.")
    st.stop()
if (data[["semantic_similarity", "rubric_coverage", "keypoint_coverage"]].min().min() < 0 or
    data[["semantic_similarity", "rubric_coverage", "keypoint_coverage"]].max().max() > 1):
    st.error("Feature scores must be between 0 and 1.")
    st.stop()
if ((data["human_marks"] < 0) | (data["human_marks"] > data["max_marks"])).any():
    st.error("Human marks must be between 0 and max_marks.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Answer records", len(data))
c2.metric("Mean maximum marks", f"{data.max_marks.mean():.1f}")
c3.metric("Feature signals", "3")

with st.expander("Preview dataset"):
    st.dataframe(data.head(15), use_container_width=True)

iterations = st.slider("Optimization iterations / generations", 20, 300, 100, 20)
seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)
if st.button("Run bio-inspired optimization", type="primary"):
    with st.spinner("Running PSO, GA and GWO..."):
        results, candidates, histories, chosen, val_scores = run_experiment(
            data, seed=int(seed), iterations=int(iterations))
    st.session_state["results"] = results
    st.session_state["candidates"] = candidates
    st.session_state["histories"] = histories
    st.session_state["chosen"] = chosen
    st.session_state["val_scores"] = val_scores

if "results" in st.session_state:
    st.header("Experimental comparison")
    st.dataframe(st.session_state["results"].round(4), use_container_width=True)
    st.success(f"Best optimizer selected using validation MAE: {st.session_state['chosen']}")
    st.subheader("Optimization convergence")
    fig, ax = plt.subplots()
    for name, history in st.session_state["histories"].items():
        ax.plot(history, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training MAE")
    ax.legend()
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    st.subheader("Optimized weights")
    weights = st.session_state["candidates"][st.session_state["chosen"]]
    weight_df = pd.DataFrame({"Feature": ["Semantic similarity", "Rubric coverage", "Key-point coverage"],
                              "Optimized weight": weights})
    st.bar_chart(weight_df.set_index("Feature"))
    st.download_button("Download comparison CSV",
                       st.session_state["results"].to_csv(index=False),
                       file_name="intelligrade_comparison.csv",
                       mime="text/csv")
    st.download_button("Download selected weights JSON",
                       pd.Series(weights, index=["semantic_similarity", "rubric_coverage", "keypoint_coverage"]).to_json(indent=2),
                       file_name="optimized_weights.json",
                       mime="application/json")

st.divider()
st.caption("Research note: optimize on training data, select on validation data, and report final metrics on an untouched test set. Repeat runs across multiple seeds for stronger evidence.")
