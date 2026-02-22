# Sri Lanka House Price Prediction

A machine learning project that predicts house prices in Sri Lanka using **XGBoost** regression, with **SHAP**-based explainability and a **Streamlit** interactive front-end.

## Dataset

- **Source**: [properties.lk](https://properties.lk/) (scraped via Supabase REST API)
- **Records**: 5,500 house-for-sale listings
- **Features**: bedrooms, bathrooms, house size, land size, district, city, negotiable status

## Project Structure

```
assignment/
  houses_data.csv            # Raw scraped dataset
  scrape.py                  # Scraping script for properties.lk
  requirements.txt           # Python dependencies
  README.md                  # This file
  report_outline.md          # Report template (convert to PDF)
  src/
    utils.py                 # Shared helpers, paths, logging
    preprocess.py            # Data cleaning, feature engineering, train/val/test split
    train.py                 # XGBoost training with Optuna hyperparameter tuning
    evaluate.py              # Metrics computation and evaluation plots
    explain.py               # SHAP explainability plots
  app/
    streamlit_app.py         # Interactive prediction UI
  models/                    # Saved model and encoders
  outputs/
    metrics.json             # Evaluation metrics
    metrics_table.csv        # Metrics in tabular form
    plots/                   # All generated charts
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Scrape Data (optional - dataset already included)

```bash
python scrape.py
```

### 3. Preprocess Data

```bash
python src/preprocess.py --input houses_data.csv --output processed.csv
```

### 4. Train Model

```bash
python src/train.py --data processed.csv
```

### 5. Evaluate Model

```bash
python src/evaluate.py
```

### 6. Generate Explanations

```bash
python src/explain.py
```

### 7. Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Methodology

| Component | Details |
|---|---|
| **Algorithm** | XGBoost Regressor |
| **Tuning** | Optuna (50 trials) with early stopping |
| **Split** | 70% train / 15% validation / 15% test |
| **Metrics** | RMSE, MAE, R-squared |
| **Explainability** | SHAP TreeExplainer (summary + dependence plots) |
| **Frontend** | Streamlit (input form + prediction + SHAP local explanation) |

## Key Results

After running the pipeline, check:
- `outputs/metrics.json` for numeric results
- `outputs/plots/` for all visualisation charts
- `report_outline.md` for the written report template
