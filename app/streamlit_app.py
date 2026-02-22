"""
app/streamlit_app.py - House Price Prediction Dashboard
========================================================
Interactive Streamlit UI that lets users enter house features and get:
    1. A predicted price (from the trained XGBoost model)
    2. A local SHAP explanation (what drove this specific prediction)
    3. Global SHAP summary plot (overall feature importance)
    4. Model performance metrics (RMSE, MAE, R2)

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib

# Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from src.utils import (
    load_model, load_encoders, load_feature_names, load_metrics,
    PLOTS_DIR, OUTPUTS_DIR, MODELS_DIR,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sri Lanka House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load assets (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_encoders():
    return load_encoders()

@st.cache_resource
def get_feature_names():
    return load_feature_names()

@st.cache_resource
def get_explainer():
    path = os.path.join(OUTPUTS_DIR, "shap_explainer.pkl")
    return joblib.load(path)

@st.cache_data
def get_metrics():
    return load_metrics()

@st.cache_data
def get_encoder_classes(_encoders):
    """Extract class lists for dropdowns."""
    result = {}
    for col, le in _encoders.items():
        result[col] = list(le.classes_)
    return result


# ---------------------------------------------------------------------------
# Sidebar - User inputs
# ---------------------------------------------------------------------------
def sidebar_inputs(encoders):
    st.sidebar.header("🏠 House Features")
    st.sidebar.markdown("Enter the details of the house you want to price:")

    classes = get_encoder_classes(encoders)

    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 8, 2)
    house_size = st.sidebar.number_input(
        "House Size (sq ft)", min_value=0, max_value=20000, value=1500, step=100
    )
    land_size = st.sidebar.number_input(
        "Land Size (perches)", min_value=0.0, max_value=500.0, value=10.0, step=0.5
    )

    district_name = st.sidebar.selectbox("District", sorted(classes.get("district", [])))
    city_name = st.sidebar.selectbox("City", sorted(classes.get("city", [])))
    negotiable = st.sidebar.checkbox("Negotiable", value=False)

    return {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "house_size_sqft": house_size,
        "land_size_perches": land_size,
        "district": district_name,
        "city": city_name,
        "negotiable": negotiable,
    }


# ---------------------------------------------------------------------------
# Build feature vector
# ---------------------------------------------------------------------------
def build_features(inputs: dict, encoders: dict, feature_names: list) -> pd.DataFrame:
    """Convert user inputs into a model-ready DataFrame."""
    row = {}
    for col in feature_names:
        if col in ("district", "city"):
            le = encoders[col]
            val = inputs[col]
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0  # fallback for unknown
        elif col == "negotiable":
            row[col] = int(inputs.get("negotiable", False))
        else:
            row[col] = inputs.get(col, 0)
    return pd.DataFrame([row], columns=feature_names)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # Header
    st.title("🏠 Sri Lanka House Price Predictor")
    st.markdown(
        "Predict house prices using **XGBoost** trained on **5,500+** listings "
        "from [properties.lk](https://properties.lk). "
        "Explanations powered by **SHAP**."
    )
    st.markdown("---")

    # Load resources
    try:
        model = get_model()
        encoders = get_encoders()
        feature_names = get_feature_names()
        explainer = get_explainer()
        metrics = get_metrics()
    except Exception as e:
        st.error(
            f"Could not load model assets. Please run the training pipeline first.\n\n"
            f"```\npython src/preprocess.py --input houses_data.csv --output processed.csv\n"
            f"python src/train.py --data processed.csv\n"
            f"python src/evaluate.py\n"
            f"python src/explain.py\n```\n\nError: {e}"
        )
        return

    # Sidebar inputs
    inputs = sidebar_inputs(encoders)

    # Build feature row
    X_input = build_features(inputs, encoders, feature_names)

    # ---- Prediction ----------------------------------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Predicted Price")
        pred = model.predict(X_input)[0]
        st.metric(
            label="Estimated Market Value",
            value=f"LKR {pred:,.0f}",
        )
        st.caption("Based on the features you entered in the sidebar.")

    with col2:
        st.subheader("Model Performance")
        test_m = metrics.get("test", {})
        st.metric("Test RMSE", f"LKR {test_m.get('rmse', 0):,.0f}")
        st.metric("Test MAE",  f"LKR {test_m.get('mae', 0):,.0f}")
        st.metric("Test R2",   f"{test_m.get('r2', 0):.4f}")

    st.markdown("---")

    # ---- Local SHAP explanation ----------------------------------------------
    st.subheader("Why this price? (SHAP Local Explanation)")
    st.markdown(
        "The chart below shows **which features pushed the prediction up or down** "
        "for *this specific house*. Red bars increase the price; blue bars decrease it."
    )

    shap_values = explainer.shap_values(X_input)
    try:
        # Waterfall plot for the single prediction
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_input.values[0],
            feature_names=feature_names,
        )
        fig_wf, ax_wf = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close("all")
    except Exception:
        # Fallback: bar plot
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        shap.plots.bar(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_input.values[0],
                feature_names=feature_names,
            ),
            show=False,
        )
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close("all")

    st.markdown("---")

    # ---- Global SHAP summary ------------------------------------------------
    st.subheader("Global Feature Impact (SHAP Summary)")
    st.markdown(
        "This plot shows **overall** which features have the most impact across "
        "all houses in the test set. Colour indicates the feature value "
        "(red = high, blue = low)."
    )
    summary_img = os.path.join(PLOTS_DIR, "shap_summary.png")
    if os.path.exists(summary_img):
        st.image(summary_img, use_container_width=True)
    else:
        st.info("Run `python src/explain.py` first to generate the summary plot.")

    # ---- Feature importance --------------------------------------------------
    st.subheader("XGBoost Feature Importance (Gain)")
    imp_img = os.path.join(PLOTS_DIR, "feature_importance.png")
    if os.path.exists(imp_img):
        st.image(imp_img, use_container_width=True)
    else:
        st.info("Run `python src/explain.py` first.")

    # ---- Evaluation plots ----------------------------------------------------
    st.subheader("Model Evaluation Plots")
    eval_col1, eval_col2 = st.columns(2)
    pa_img = os.path.join(PLOTS_DIR, "predicted_vs_actual.png")
    rh_img = os.path.join(PLOTS_DIR, "residual_histogram.png")

    with eval_col1:
        if os.path.exists(pa_img):
            st.image(pa_img, caption="Predicted vs Actual", use_container_width=True)
    with eval_col2:
        if os.path.exists(rh_img):
            st.image(rh_img, caption="Residual Distribution", use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption("Built for ML & Pattern Recognition Assignment | Data from properties.lk")


if __name__ == "__main__":
    main()
