# 🏥 Medical Cost Personal Insurance Prediction

Predicting individual medical insurance charges from demographic and lifestyle attributes, with a focus on the smoking × BMI interaction that drives the largest cost differences.

**Suggested repo name:** `medical-cost-insurance-prediction`

## Overview

Health insurers need to price policies accurately based on expected claim costs. This project predicts an individual's annual medical insurance charges from six attributes — age, sex, BMI, number of children, smoking status, and region — and quantifies which factors drive cost the most.

## Dataset

- **Source:** "Medical Cost Personal Datasets" (mirrored via [stedy/Machine-Learning-with-R-datasets](https://github.com/stedy/Machine-Learning-with-R-datasets))
- **Size:** 1,338 records, 7 columns
- **Target:** `charges` (USD, continuous, right-skewed)
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`

## Repository Structure

```
medical-cost-insurance-prediction/
├── README.md
├── requirements.txt
└── notebooks/
    └── 03_Medical_Cost_Insurance_Prediction.ipynb
```

## Getting Started

```bash
git clone https://github.com/<your-username>/medical-cost-insurance-prediction.git
cd medical-cost-insurance-prediction
pip install -r requirements.txt
jupyter notebook notebooks/03_Medical_Cost_Insurance_Prediction.ipynb
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

## Methodology

1. **EDA** — charge distributions (raw & log), charges by smoker/BMI/age/region/sex, correlation heatmap.
2. **Feature engineering** — added `bmi_smoker` and `age_smoker` interaction terms and an `is_obese` (BMI ≥ 30) flag to capture the strong smoking × BMI interaction visible in EDA.
3. **Preprocessing** — `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) in a scikit-learn `Pipeline`.
4. **Model comparison** — Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost compared via 5-fold CV.
5. **Tuning** — `GridSearchCV` over Gradient Boosting (n_estimators, max_depth, learning_rate).
6. **Evaluation** — RMSE/MAE/R² on a held-out 20% test set, residual plots, feature importance.

## Results

| Metric | Value |
|---|---|
| Final model | **Gradient Boosting** (200 trees, depth=2, lr=0.03) |
| **Test RMSE** | **$4,370.70** |
| Test MAE | $2,507.10 |
| **Test R²** | **0.877** |

**Top features:** `bmi_smoker` (interaction), `smoker`, `age_smoker` — smoking status and its interaction with BMI dominate all other factors.

## Key Insights

- Smokers pay roughly 3–4× more on average than non-smokers.
- The smoking × obesity combination is the single costliest profile — BMI alone (for non-smokers) has a much weaker effect than BMI combined with smoking.
- Sex and region contribute comparatively little.

## Future Work

- Quantile regression to model the cost *distribution*, not just the mean (useful for high-cost outlier cases).
- Incorporate actual diagnosis/claims history if available.
- Model calibration specifically for the high-cost tail, where insurers care most about accuracy.

## Data Source & License

Dataset commonly distributed as a companion to *Machine Learning with R* (Brett Lantz); used here for educational/portfolio purposes.

## License

MIT
