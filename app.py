import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Insurance Premium Estimator", layout="wide")

# -----------------------------
# Paths / loading
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "best_random_forest_model.pkl"

def load_model(path) -> object:
    path = Path(path)
    if not path.exists():
        st.error(f"Model file '{path.name}' not found at {path}. Place it next to app.py.")
        return None
    try:
        model = joblib.load(path)
        st.success(f"Loaded model: {path.name}")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def get_feature_names(model) -> list:
    """Return the exact feature names used at fit time, even if model is a Pipeline."""
    # Plain estimator with feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # Pipeline case
    if hasattr(model, "named_steps"):
        # try common regressor step names
        for step_name in ["randomforestregressor", "gradientboostingregressor",
                          "linearregression", "mlpregressor"]:
            step = model.named_steps.get(step_name)
            if step is not None and hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
        # try any step that has feature_names_in_
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    # Fallback: none found
    return None

# -----------------------------
# App
# -----------------------------
st.title("Insurance Premium Estimator")

with st.form("single_pred_v1"):
    st.subheader("Enter details")
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        age = st.number_input("Age", min_value=18, max_value=66, value=45, step=1)
        surgeries = st.number_input("NumberOfMajorSurgeries", min_value=0, max_value=3, value=0, step=1)

    with c2:
        height_cm = st.number_input("Height (cm)", min_value=145, max_value=188, value=170, step=1)
        weight_kg = st.number_input("Weight (kg)", min_value=51, max_value=132, value=68, step=1)
        bmi = round(float(weight_kg) / ((float(height_cm) / 100.0) ** 2), 2)
        st.caption(f"Computed BMI: **{bmi}**")

    with c3:
        diabetes    = st.checkbox("Diabetes", value=False)
        bpp         = st.checkbox("BloodPressureProblems", value=False)
        transplant  = st.checkbox("AnyTransplants", value=False)
        chronic     = st.checkbox("AnyChronicDiseases", value=False)
        allergy     = st.checkbox("KnownAllergies", value=False)
        cancerfam   = st.checkbox("HistoryOfCancerInFamily", value=False)

    submitted = st.form_submit_button("Predict Insurance")

if submitted:
    model = load_model(MODEL_PATH)
    if model is None:
        st.stop()

    FEATURES = get_feature_names(model)
    if not FEATURES:
        st.error("Could not read feature names from the model. Re‑save the pickle with scikit‑learn ≥1.0 so it includes `feature_names_in_`.")
        st.stop()

    # Building a row that matches the model's exact training columns
    values = {}
    # First, creating a dict of all UI values we can map from
    ui = {
        "Age": int(age),
        "Diabetes": int(diabetes),
        "BloodPressureProblems": int(bpp),
        "AnyTransplants": int(transplant),
        "AnyChronicDiseases": int(chronic),
        "KnownAllergies": int(allergy),
        "HistoryOfCancerInFamily": int(cancerfam),
        "NumberOfMajorSurgeries": int(surgeries),
        "Height": float(height_cm),         
        "Weight": float(weight_kg),         
        "BMI": float(bmi),                  
    }

    # Now filling required features in the exact order
    missing = []
    for f in FEATURES:
        if f in ui:
            values[f] = ui[f]
        else:
            # allowing unseen columns to default to 0 if they are binary-like; otherwise mark missing
            missing.append(f)

    if missing:
        st.error(f"The model expects columns not present in the UI: {missing}. "
                 "Either retrain with matching features or add these inputs.")
        st.stop()

    X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

    try:
        yhat = model.predict(X)[0]
        st.metric("Estimated Premium", f"{yhat:,.0f}")   # no currency symbol
        with st.expander("Inputs used"):
            st.dataframe(X, use_container_width=True)
        st.caption("Features used (in order): " + ", ".join(FEATURES))
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Tip: This app auto-detects whether the model expects BMI or Height/Weight and builds inputs accordingly.")

