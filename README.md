# Customer Churn Prediction — Telecom

End-to-end machine learning pipeline predicting which telecom customers 
are likely to churn, enabling proactive retention interventions before 
revenue is lost.

## Problem Statement
A telecom company is losing customers to competitors. Every churned 
customer represents lost future revenue and a costly re-acquisition 
problem — replacing a customer costs 5-7x more than retaining one. 
This model identifies high-risk customers before they leave, giving 
the retention team a weekly actionable target list.

## Dataset
**IBM Telco Customer Churn Dataset**
- 7,043 customer records, 21 features, 1 binary target
- Features cover demographics, account info, services subscribed, 
  and billing details
- Source: IBM Sample Data

## Pre-EDA Hypothesis
Before any analysis, I hypothesised that tenure, contract type, 
and monthly charges would be the primary churn drivers — based on 
domain reasoning about telecom customer behaviour. All three were 
confirmed by the data and validated by the model's feature importance.

## Key Steps

**1. Data Quality Investigation**
Identified TotalCharges stored as text due to blank values for 
brand-new customers (tenure = 0). Diagnosed root cause before 
fixing — filled with 0 rather than median, because these customers 
genuinely had zero accumulated charges, not missing measurements.

**2. Exploratory Data Analysis**
- Month-to-month customers churn at 42.7% vs 2.8% for two-year contracts
- Churned customers had median tenure of 10 months vs 38 months retained
- Fiber optic customers churn at 41.9% despite being the premium service
- Churned customers paid median $79.65/month vs $64.43 for retained

**3. Encoding Strategy**
Applied three different encoding approaches based on data type:
- Binary encoding (Yes/No columns)
- Ordinal encoding (Contract — genuine commitment progression)
- One-hot encoding (multi-category columns with no natural order)

**4. Preprocessing**
Stratified train-test split, StandardScaler, SMOTE oversampling 
on training data only to address 73.5/26.5 class imbalance.

**5. Modelling**
Trained and compared Logistic Regression, Random Forest, and XGBoost.

**6. Evaluation**
Prioritised Recall as primary metric — missing a churning customer 
(false negative) costs the business their entire future revenue, 
while a false positive only wastes a retention call.

## Results

| Model | Accuracy | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.738 | **0.799** | 0.504 | 0.841 |
| Random Forest | 0.750 | 0.781 | 0.520 | 0.842 |
| XGBoost | 0.776 | 0.671 | 0.567 | 0.842 |

Logistic Regression selected as final model — highest Recall despite 
being the simplest model, demonstrating that complexity doesn't always 
win when the right metric guides selection.

## Key Insights
- Contract type is the dominant churn driver (importance 0.32, 
  2.5x higher than the next feature)
- Over 55% of customers are on month-to-month plans — the highest 
  churn risk segment
- Fiber optic customers churn at 41.9% — a value perception problem, 
  not a pricing problem
- Electronic check payment signals higher churn risk than automatic 
  payment methods

## Ethical Considerations
Project addresses demographic fairness in model features, retention 
offer equity across customer segments, data privacy compliance, 
model transparency for retention agents, and drift monitoring 
requirements. Full discussion in notebook.

## Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn 
(SMOTE) · SHAP · Matplotlib · Seaborn · Jupyter Notebook

## Author
Astha Singh Kushwaha
[LinkedIn](https://linkedin.com/in/astha-singh-k) · [GitHub](https://github.com/AsthaSK)
