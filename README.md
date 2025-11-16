# Customer Churn Prediction using CatBoost + Streamlit

This project predicts whether a customer is likely to churn using the 
**Churn_Modelling.csv** dataset. The model is built using:

- CatBoost (tuned with 800 iterations, depth 8)
- Feature engineering
- Threshold optimization (final threshold = 0.35)
- StandardScaler
- Streamlit app for deployment

---

## Project Highlights

### Advanced Feature Engineering
Engineered features significantly improved accuracy and recall:
- Balance-to-Salary Ratio
- Products Per Tenure
- Age-to-Tenure Ratio
- Senior Citizen Flag
- Has Balance Flag
- Credit Score Binning

### Tuned CatBoost Model
Achieved:

| Metric | Score |
|-------|-------|
| Accuracy | 78% |
| Recall (Churn class) | **76%** |
| F1 Score | 58% |

### ✔ Optimized Decision Threshold
Default 0.5 → **0.35** to improve churn detection.

This increased churn recall from 0.65 → **0.76**.

---

## Streamlit Web App

The deployed app:
- Takes customer inputs
- Applies feature engineering
- Scales inputs using the saved scaler
- Predicts churn probability
- Classifies risk using threshold 0.35
- Provides interpretation + recommendations

Run the app:

```bash
streamlit run streamlit_app/app.py
