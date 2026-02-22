# Machine Learning Assignment Report
## Predicting House Prices in Sri Lanka using XGBoost

**Module**: Machine Learning & Pattern Recognition
**Dataset**: 5,500 house-for-sale listings from [properties.lk](https://properties.lk)

---

## 1. Introduction

- **Problem**: Predict house sale prices in Sri Lanka from listing attributes (bedrooms, bathrooms, size, location, etc.)
- **Task type**: Supervised regression
- **Target variable**: `price` (in LKR)
- **Dataset source**: properties.lk -- real estate listings scraped via the site's Supabase REST API
- **Dataset size**: 5,500 raw records, ~5,200 after outlier removal

---

## 2. Selection of Machine Learning Algorithm

### 2.1 Why XGBoost?

| Criterion | XGBoost | Linear Regression | Decision Tree |
|---|---|---|---|
| Non-linear relationships | Yes (ensemble of trees) | No (assumes linearity) | Yes (single tree) |
| Overfitting control | Strong (regularisation, early stopping) | Weak | Weak (prone to overfitting) |
| Missing value handling | Native support | Requires imputation | Limited |
| Feature importance | Built-in (gain/cover/weight) | Coefficients only | Built-in but noisy |
| SHAP compatibility | TreeExplainer (exact, fast) | LinearExplainer | TreeExplainer |
| Ensemble learning | Yes (gradient boosting) | No | No |

### 2.2 How XGBoost Differs from Basic Models

1. **Gradient Boosting**: Builds trees sequentially -- each new tree corrects the errors of the previous ensemble, unlike a single Decision Tree or independent Random Forest trees.
2. **Regularisation**: L1 (`reg_alpha`) and L2 (`reg_lambda`) penalties shrink leaf weights, preventing overfitting (absent in basic Decision Trees).
3. **Early Stopping**: Training halts automatically when validation performance stops improving, avoiding wasted computation.
4. **Histogram-based splitting**: Bins continuous features for faster split-finding (O(n) instead of O(n log n)).

---

## 3. Data Collection & Preprocessing

### 3.1 Data Scraping

- **Source**: properties.lk Supabase REST API (public anonymous key)
- **Method**: Python `requests` library, paginated API calls (100 records/batch)
- **Delays**: 1-second pause between requests (ethical scraping)
- **Output**: `houses_data.csv` (5,500 rows x 14 columns)

### 3.2 Feature Engineering

| Feature | Transformation |
|---|---|
| `house_size` | Parsed from "3000 sq ft" strings to numeric `house_size_sqft`; blanks filled with median |
| `land_size` | Parsed from "10.5 Perches" to numeric `land_size_perches`; blanks filled with median |
| `district`, `city` | Label-encoded (integer mapping) |
| `negotiable` | Boolean to integer (0/1) |
| `price` | Outliers removed using IQR method (1st-99th percentile bounds) |

### 3.3 Dropped Columns

`id`, `title`, `description`, `posted_date`, `address`, `price_type` -- these are either identifiers, free text, or non-predictive metadata.

---

## 4. Model Training & Evaluation

### 4.1 Data Splitting

| Split | Proportion | Random State |
|---|---|---|
| Training | 70% | 42 |
| Validation | 15% | 42 |
| Test | 15% | 42 |

### 4.2 Hyperparameter Tuning

- **Method**: Optuna (Bayesian optimisation with Tree-structured Parzen Estimator)
- **Trials**: 50
- **Objective**: Minimise validation RMSE
- **Early stopping**: 50 rounds (patience) on validation set

**Tuned hyperparameters:**

| Parameter | Search Range |
|---|---|
| `n_estimators` | 200 -- 1500 |
| `max_depth` | 3 -- 10 |
| `learning_rate` | 0.01 -- 0.3 (log scale) |
| `subsample` | 0.5 -- 1.0 |
| `colsample_bytree` | 0.5 -- 1.0 |
| `reg_alpha` | 1e-8 -- 10 (log scale) |
| `reg_lambda` | 1e-8 -- 10 (log scale) |
| `min_child_weight` | 1 -- 10 |

### 4.3 Performance Metrics

| Split | RMSE (LKR) | MAE (LKR) | R-squared |
|---|---|---|---|
| Train | 33,780,948 | 16,152,195 | 0.6592 |
| Validation | 42,768,276 | 18,405,824 | 0.4832 |
| Test | 44,699,152 | 19,269,856 | 0.4400 |

### 4.4 Evaluation Plots


1. **Predicted vs Actual Scatter** (`predicted_vs_actual.png`): Points near the diagonal indicate accurate predictions.
2. **Residual Histogram** (`residual_histogram.png`): A symmetric distribution centred at zero indicates unbiased predictions.

---

## 5. Explainability

### 5.1 SHAP Analysis (Mandatory)

- **Method**: `shap.TreeExplainer` -- computes exact Shapley values for tree-based models in polynomial time.
- **Why SHAP?** Provides both global (which features matter overall) and local (why was this specific prediction made) explanations, grounded in cooperative game theory.

**SHAP Summary Plot** (`shap_summary.png`):
- Each dot = one data point
- X-axis = SHAP value (impact on predicted price)
- Colour = feature value (red = high, blue = low)
- Top features = most influential overall

**SHAP Dependence Plot** (`shap_dependence.png`):
- Shows how the top feature's value relates to its impact on price
- Colour shows interaction with the most correlated feature

### 5.2 Feature Importance (Additional)

**XGBoost Gain-based Importance** (`feature_importance.png`):
- Gain = average reduction in loss when a feature is used for splitting
- Higher gain = more useful feature

---

## 6. Bonus: Streamlit Front-End

- **Input form**: Sidebar with sliders/dropdowns for bedrooms, bathrooms, house size, land size, district, city, negotiable
- **Outputs**:
  - Predicted price (LKR)
  - Local SHAP waterfall plot (what drove this specific prediction)
  - Global SHAP summary (overall feature importance)
  - Model metrics (RMSE, MAE, R2)

**Launch command**: `streamlit run app/streamlit_app.py`

---

## 7. Conclusion

- XGBoost effectively captures non-linear relationships in Sri Lankan house pricing data
- Key price drivers identified via SHAP: `house_size_sqft` (top feature), followed by `land_size_perches`, `city`, `district`, and `bedrooms`
- The model can serve as a practical tool for estimating house prices given basic listing attributes
- Limitations: relies on listing data quality; does not capture neighbourhood-level factors or market trends over time

---

## Appendix

### A. Project Structure
Refer to `README.md` for the complete file layout and run commands.

### B. How to Reproduce
```bash
pip install -r requirements.txt
python src/preprocess.py --input houses_data.csv --output processed.csv
python src/train.py --data processed.csv
python src/evaluate.py
python src/explain.py
streamlit run app/streamlit_app.py
```

### C. References
1. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
2. Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. NIPS.
3. Akiba, T. et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework.
