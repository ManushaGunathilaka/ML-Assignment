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
    load_model, load_encoders, load_feature_names, load_metrics, load_transform_info,
    PLOTS_DIR, OUTPUTS_DIR, MODELS_DIR,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sri Lanka House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for better UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d5a87 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSelectbox label {
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Price display card */
    .price-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }
    .price-card h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .price-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-card h4 {
        margin: 0;
        color: #495057;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .metric-card h3 {
        margin: 0.5rem 0 0 0;
        color: #212529;
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    /* Section headers */
    .section-header {
        background: #f1f3f4;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #667eea;
    }
    .section-header h3 {
        margin: 0;
        color: #1e3a5f;
        font-size: 1.1rem;
    }
    
    /* Feature summary pills */
    .feature-pill {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

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
    st.sidebar.markdown("## 🏠 Property Details")
    st.sidebar.markdown("---")
    
    classes = get_encoder_classes(encoders)

    # Location Section
    st.sidebar.markdown("### 📍 Location")
    district_name = st.sidebar.selectbox(
        "District",
        sorted(classes.get("district", [])),
        help="Select the district where the property is located"
    )
    city_name = st.sidebar.selectbox(
        "City",
        sorted(classes.get("city", [])),
        help="Select the city or area"
    )
    
    st.sidebar.markdown("---")
    
    # Property Features Section
    st.sidebar.markdown("### 🏗️ Property Features")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        bedrooms = st.number_input(
            "🛏️ Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Number of bedrooms"
        )
    with col2:
        bathrooms = st.number_input(
            "🚿 Bathrooms",
            min_value=1,
            max_value=8,
            value=2,
            step=1,
            help="Number of bathrooms"
        )
    
    house_size = st.sidebar.number_input(
        "📐 House Size (sq ft)",
        min_value=100,
        max_value=20000,
        value=1500,
        step=50,
        help="Total built-up area in square feet"
    )
    
    land_size = st.sidebar.number_input(
        "🌳 Land Size (perches)",
        min_value=1.0,
        max_value=500.0,
        value=10.0,
        step=0.5,
        format="%.1f",
        help="Total land area in perches (1 perch = 25.29 sq m)"
    )
    
    st.sidebar.markdown("---")
    
    # Additional Options
    st.sidebar.markdown("### ⚙️ Options")
    negotiable = st.sidebar.toggle(
        "💬 Price Negotiable",
        value=False,
        help="Is the seller open to price negotiation?"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Adjust the values above to get an instant price estimate")

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
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="color: #1e3a5f; margin-bottom: 0.5rem;">🏠 Sri Lanka House Price Predictor</h1>
        <p style="color: #666; font-size: 1.1rem;">
            AI-powered price estimation using <b>XGBoost</b> trained on <b>5,500+</b> real listings
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load resources
    try:
        model = get_model()
        encoders = get_encoders()
        feature_names = get_feature_names()
        explainer = get_explainer()
        metrics = get_metrics()
    except Exception as e:
        st.error(
            f"⚠️ Could not load model assets. Please run the training pipeline first.\n\n"
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
    
    # Make prediction (apply inverse log transform if model was trained with log)
    pred_raw = model.predict(X_input)[0]
    transform_info = load_transform_info()
    if transform_info.get("log_transform", False):
        pred = np.expm1(pred_raw)  # inverse of log1p
    else:
        pred = pred_raw

    # ---- Main Content Area ---------------------------------------------------
    
    # Property Summary + Price Card
    col_summary, col_price = st.columns([1.5, 1])
    
    with col_summary:
        st.markdown("### 📋 Your Property Summary")
        
        # Display input summary as styled pills
        summary_html = f"""
        <div style="padding: 1rem; background: #f8f9fa; border-radius: 12px; margin-bottom: 1rem;">
            <span class="feature-pill">📍 {inputs['district']}, {inputs['city']}</span>
            <span class="feature-pill">🛏️ {inputs['bedrooms']} Beds</span>
            <span class="feature-pill">🚿 {inputs['bathrooms']} Baths</span>
            <span class="feature-pill">📐 {inputs['house_size_sqft']:,} sq ft</span>
            <span class="feature-pill">🌳 {inputs['land_size_perches']:.1f} perches</span>
            <span class="feature-pill">{'💬 Negotiable' if inputs['negotiable'] else '🔒 Fixed Price'}</span>
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)
    
    with col_price:
        # Price Card with gradient
        st.markdown(f"""
        <div class="price-card">
            <p style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">Estimated Value</p>
            <h2>LKR {pred:,.0f}</h2>
            <p style="font-size: 0.85rem; margin-top: 0.75rem;">
                ≈ Rs. {pred/1000000:.2f} Million
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🔍 Price Explanation", "📊 Model Insights", "📈 Performance"])
    
    # ---- Tab 1: Price Explanation --------------------------------------------
    with tab1:
        st.markdown("""
        <div class="section-header">
            <h3>Why This Price? - SHAP Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        The chart below shows **which features influenced the predicted price** for your specific property.
        - **Red/Positive bars** → Features that **increased** the price
        - **Blue/Negative bars** → Features that **decreased** the price
        """)

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
        
        # Top factors explanation
        abs_shap = np.abs(shap_values[0])
        top_idx = np.argsort(abs_shap)[::-1][:3]
        
        st.markdown("#### 🎯 Top 3 Factors Affecting This Price")
        
        # Check if log transform was used - SHAP values are in log scale
        transform_info = load_transform_info()
        use_log = transform_info.get("log_transform", False)
        
        factor_cols = st.columns(3)
        for i, idx in enumerate(top_idx):
            feature = feature_names[idx]
            impact = shap_values[0][idx]
            direction = "📈 Increases" if impact > 0 else "📉 Decreases"
            
            if use_log:
                # Convert log-scale SHAP to approximate percentage impact
                # exp(shap_value) - 1 gives approximate % change
                pct_impact = (np.exp(abs(impact)) - 1) * 100
                with factor_cols[i]:
                    st.metric(
                        label=feature.replace("_", " ").title(),
                        value=f"{pct_impact:.1f}%",
                        delta=direction + " price",
                        delta_color="normal" if impact > 0 else "inverse"
                    )
            else:
                with factor_cols[i]:
                    st.metric(
                        label=feature.replace("_", " ").title(),
                        value=f"LKR {abs(impact):,.0f}",
                        delta=direction + " price",
                        delta_color="normal" if impact > 0 else "inverse"
                    )

    # ---- Tab 2: Model Insights -----------------------------------------------
    with tab2:
        st.markdown("""
        <div class="section-header">
            <h3>Global Feature Importance</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_shap, col_imp = st.columns(2)
        
        with col_shap:
            st.markdown("#### SHAP Summary Plot")
            st.caption("Shows overall feature impact across all properties. Red = high value, Blue = low value.")
            summary_img = os.path.join(PLOTS_DIR, "shap_summary.png")
            if os.path.exists(summary_img):
                st.image(summary_img)
            else:
                st.info("Run `python src/explain.py` to generate this plot.")
        
        with col_imp:
            st.markdown("#### XGBoost Feature Importance")
            st.caption("Built-in importance based on gain in tree splits.")
            imp_img = os.path.join(PLOTS_DIR, "feature_importance.png")
            if os.path.exists(imp_img):
                st.image(imp_img)
            else:
                st.info("Run `python src/explain.py` to generate this plot.")
        
        # Dependence plot
        dep_img = os.path.join(PLOTS_DIR, "shap_dependence_top_feature.png")
        if os.path.exists(dep_img):
            st.markdown("#### SHAP Dependence Plot (Top Feature)")
            st.caption("Shows how the top feature affects predictions at different values.")
            st.image(dep_img)

    # ---- Tab 3: Model Performance --------------------------------------------
    with tab3:
        st.markdown("""
        <div class="section-header">
            <h3>Model Performance Metrics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics cards
        test_m = metrics.get("test", {})
        val_m = metrics.get("val", {})
        
        st.markdown("#### Test Set Performance")
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            st.markdown("""
            <div class="metric-card">
                <h4>Root Mean Square Error</h4>
                <h3>LKR {:,.0f}</h3>
            </div>
            """.format(test_m.get('rmse', 0)), unsafe_allow_html=True)
        
        with metric_cols[1]:
            st.markdown("""
            <div class="metric-card">
                <h4>Mean Absolute Error</h4>
                <h3>LKR {:,.0f}</h3>
            </div>
            """.format(test_m.get('mae', 0)), unsafe_allow_html=True)
        
        with metric_cols[2]:
            r2_val = test_m.get('r2', 0)
            r2_pct = r2_val * 100
            st.markdown(f"""
            <div class="metric-card">
                <h4>R² Score</h4>
                <h3>{r2_val:.4f} ({r2_pct:.1f}%)</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Evaluation plots
        st.markdown("#### Model Evaluation Plots")
        eval_col1, eval_col2 = st.columns(2)
        pa_img = os.path.join(PLOTS_DIR, "predicted_vs_actual.png")
        rh_img = os.path.join(PLOTS_DIR, "residual_histogram.png")

        with eval_col1:
            if os.path.exists(pa_img):
                st.image(pa_img, caption="Predicted vs Actual Prices")
            else:
                st.info("Run `python src/evaluate.py` to generate this plot.")
        
        with eval_col2:
            if os.path.exists(rh_img):
                st.image(rh_img, caption="Residual Distribution")
            else:
                st.info("Run `python src/evaluate.py` to generate this plot.")
        
        # Metrics comparison table
        if val_m:
            st.markdown("#### Train/Validation/Test Comparison")
            metrics_df = pd.DataFrame({
                "Metric": ["RMSE (LKR)", "MAE (LKR)", "R² Score"],
                "Validation": [
                    f"{val_m.get('rmse', 0):,.0f}",
                    f"{val_m.get('mae', 0):,.0f}",
                    f"{val_m.get('r2', 0):.4f}"
                ],
                "Test": [
                    f"{test_m.get('rmse', 0):,.0f}",
                    f"{test_m.get('mae', 0):,.0f}",
                    f"{test_m.get('r2', 0):.4f}"
                ]
            })
            st.dataframe(metrics_df, hide_index=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; color: #888;">
        <p>🎓 Built for ML & Pattern Recognition Assignment | 📊 Data from 
        <a href="https://properties.lk" target="_blank" style="color: #667eea;">properties.lk</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
