# Credit Risk Modeling with Explainable AI

A comprehensive machine learning project for predicting credit default risk using multiple classification algorithms, with a focus on model interpretability through explainable AI (XAI) techniques.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Explainability](#explainability)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Overview

This project addresses the critical problem of credit risk assessment by building predictive models to identify customers likely to default on their credit obligations. The implementation includes:

- Three different classification algorithms: Logistic Regression, XGBoost, and Random Forest
- Comprehensive data preprocessing and feature engineering
- Threshold optimization for improved business metrics
- Explainability analysis using SHAP and LIME
- Comparative model evaluation

## Dataset

The project uses the "Give Me Some Credit" dataset containing information about credit borrowers.

**Features:**
- `SeriousDlqin2yrs`: Target variable (1 = defaulted, 0 = no default)
- `RevolvingUtilizationOfUnsecuredLines`: Total balance on credit cards divided by credit limits
- `age`: Age of borrower
- `NumberOfTime30-59DaysPastDueNotWorse`: Number of times 30-59 days past due
- `DebtRatio`: Monthly debt payments divided by monthly income
- `MonthlyIncome`: Monthly income of borrower
- `NumberOfOpenCreditLinesAndLoans`: Number of open loans and credit lines
- `NumberOfTimes90DaysLate`: Number of times 90+ days past due
- `NumberRealEstateLoansOrLines`: Number of real estate loans
- `NumberOfTime60-89DaysPastDueNotWorse`: Number of times 60-89 days past due
- `NumberOfDependents`: Number of dependents

**Dataset Statistics:**
- Training samples: Approximately 150,000 records
- Test split: 20% of data
- Class imbalance: Minority class (defaults) comprises roughly 6-7% of observations

## Project Structure

```
credit-risk-modeling/
│
├── credit_risk_modelling.py    # Main implementation file
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── data/
    └── cs-training.csv         # Dataset file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-risk-modeling.git
cd credit-risk-modeling
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Methodology

### 1. Data Preprocessing

**Missing Value Imputation:**
- Monthly income: Filled with median value from training set
- Number of dependents: Filled with mode from training set

**Outlier Treatment:**
- Count features (late payments, dependents): Capped at 10
- Continuous features: Capped at 99.5th percentile to reduce extreme outliers

**Feature Engineering:**
- Applied log transformation (log1p) to highly skewed features: MonthlyIncome, DebtRatio, and UnsecLines
- Created new log-transformed features and dropped original skewed columns

**Scaling:**
- StandardScaler applied to all features for Logistic Regression
- Scaled data also used for XGBoost and Random Forest for consistency

### 2. Train-Test Split

- Split ratio: 80% training, 20% testing
- Stratified sampling to maintain class distribution
- Random state: 42 for reproducibility

### 3. Class Imbalance Handling

- Logistic Regression: `class_weight='balanced'`
- XGBoost: `scale_pos_weight` calculated based on class ratio
- Random Forest: `class_weight='balanced'`

## Models

### Logistic Regression
- Solver: liblinear
- Balanced class weights for handling imbalance
- Interpretable coefficients and odds ratios

### XGBoost Classifier
- Objective: binary logistic
- Evaluation metric: AUC
- Number of estimators: 150
- Learning rate: 0.1
- Scale positive weight calculated from class imbalance ratio

### Random Forest Classifier
- Number of estimators: 100
- Max depth: 10
- Balanced class weights
- Parallel processing enabled (n_jobs=-1)

## Results

### Model Performance Comparison

All models were evaluated using multiple metrics:

**Evaluation Metrics:**
- AUC-ROC: Area under the Receiver Operating Characteristic curve
- Precision-Recall AUC: Area under the Precision-Recall curve
- F1 Score: Harmonic mean of precision and recall
- Optimal Threshold: Determined by maximizing F1 score

**Key Findings:**
- All models achieved strong AUC scores above 0.85
- XGBoost demonstrated the best overall performance
- Optimal thresholds were identified for each model, typically lower than the default 0.5 to better capture defaults
- F2 score optimization was also explored for XGBoost to emphasize recall

### Feature Importance

**Most Important Features (Consistent Across Models):**
1. Late payment indicators (Late90, Late3059, Late6089)
2. Revolving credit utilization (UnsecLines_log)
3. Age
4. Debt ratio (DebtRatio_log)
5. Number of open credit lines

The project includes consolidated feature importance visualizations comparing all three models.

## Explainability

### SHAP (SHapley Additive exPlanations)

**Global Interpretability:**
- Summary plots showing feature importance across all predictions
- Generated for both XGBoost and Random Forest models
- Sample size: 1,000-1,500 instances for computational efficiency
- Spearman rank correlation used to assess consistency between models

**Dependence Plots:**
- Interaction analysis between key features
- Example: UnsecLines_log vs. Late3059 showing how features interact to influence predictions

### LIME (Local Interpretable Model-agnostic Explanations)

**Instance-Level Explanations:**
Local explanations provided for four key prediction scenarios:
1. True Positive: Correctly predicted default
2. True Negative: Correctly predicted non-default
3. False Positive: Incorrectly predicted as default
4. False Negative: Missed default prediction

**LIME Configuration:**
- Number of features explained: 8 top contributors
- Kernel width: 0.75
- Training data used as background distribution

## Usage

### Running the Full Pipeline

```python
# Load and preprocess data
df = pd.read_csv('data/cs-training.csv')

# Run the complete analysis
python credit_risk_modelling.py
```

### Making Predictions on New Data

```python
# Example for XGBoost model
import pickle
import pandas as pd

# Load trained model (you'll need to save it first)
# model_xgb.save_model('xgboost_model.json')

# Prepare new data with same preprocessing
new_data = preprocess_new_data(raw_data)

# Make predictions
predictions = model_xgb.predict_proba(new_data)[:, 1]

# Apply optimal threshold
final_predictions = (predictions >= best_threshold_xgb).astype(int)
```

### Generating Explanations

```python
# SHAP explanation for a single instance
import shap

explainer = shap.Explainer(model_xgb, X_sample)
shap_values = explainer(instance_to_explain)
shap.plots.waterfall(shap_values[0])

# LIME explanation
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled.values,
    feature_names=X_train_scaled.columns.tolist(),
    class_names=['No Default', 'Default'],
    mode='classification'
)

exp = explainer.explain_instance(
    data_row=instance.values,
    predict_fn=model_xgb.predict_proba,
    num_features=8
)
exp.show_in_notebook()
```

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
shap>=0.41.0
lime>=0.2.0
scipy>=1.7.0
```

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Key Insights

1. **Late payment history is the strongest predictor**: Features tracking past due payments (30-59, 60-89, and 90+ days) are consistently the most important across all models.

2. **Threshold optimization is critical**: The default classification threshold of 0.5 is suboptimal for imbalanced credit risk data. Optimal thresholds were found to be lower, improving F1 scores significantly.

3. **Model consistency**: High Spearman correlation between SHAP importance rankings from XGBoost and Random Forest suggests robust feature importance identification.

4. **Explainability enhances trust**: LIME and SHAP provide complementary views - LIME for individual loan decisions, SHAP for overall model behavior.

## Future Enhancements

- Implement ensemble methods combining all three models
- Add temporal validation for time-series credit data
- Incorporate additional external data sources
- Deploy model as a REST API for real-time predictions
- Add monitoring for model drift and performance degradation
- Implement fairness metrics to ensure unbiased predictions

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: "Give Me Some Credit" competition dataset
- SHAP library: Scott Lundberg and the SHAP contributors
- LIME library: Marco Tulio Ribeiro and contributors

## Contact

For questions or collaboration opportunities, please open an issue in the repository.

---

**Note:** This project is for educational and research purposes. Credit risk assessment in production environments requires additional considerations including regulatory compliance, fairness testing, and ongoing model monitoring.