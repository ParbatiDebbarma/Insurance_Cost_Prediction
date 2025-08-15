# Insurance Cost Prediction

A complete ML workflow to predict health insurance premium prices and a Streamlit web app for single-person estimates.

## 1) Problem, Data & Target
- **Goal:** Estimate `PremiumPrice` from individual health profile.
- **Features used:** Age, BMI (or Height & Weight → BMI), Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries.
- **Target metric:** R² (primary), with RMSE/MAE as error metrics.

## 2) Workflow (what’s in the notebook)
- **EDA:** distributions, outliers, pairwise relationships.
- **Hypothesis testing:** e.g., presence of chronic disease/diabetes increases average premium (t‑tests/ANOVA).
- **Feature engineering:** computed BMI from height & weight.
- **Models tried:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting, MLP.
- **Evaluation:** 5‑fold CV with R² / RMSE / MAE; model comparison table.

## 3) Final Scores (5‑fold means)
| Model | RMSE | MAE | R² | Notes |
|---|---:|---:|---:|---|
| Linear Regression | 3850.05 | 2746.90 | 0.608 | Interpretable, stable |
| Decision Tree | 4228.35 | 1743.82 | 0.500 | High variance |
| **Random Forest (final)** | **2829.00** | **1114.40** | **0.785** | Best accuracy, handles non‑linearities |
| Gradient Boosting | 3305.16 | 1866.30 | 0.709 | Stable, slightly worse than RF |
| MLP | 3950.15 | 2829.37 | 0.587 | Underperformed here |

> Final choice: **RandomForestRegressor** trained on df.  
> The Streamlit app accepts Height/Weight and computes BMI internally when needed.

## 4) Streamlit App (deployment files)
- `app.py` – single‑person premium estimator (dark theme, checkboxes for binaries).
- `requirements.txt` – Python deps.
- `.streamlit/config.toml` – dark theme.
- `best_random_forest_model.pkl` – trained model (kept small; no PII).



---


## 5) Deployment Steps

Deployed the application on **Streamlit Community Cloud** by preparing the project folder with all required files (`app.py`, `requirements.txt`, `.streamlit/config.toml`, `best_random_forest_model.pkl`, and the project notebook) and pushing them to a public GitHub repository. The repository was then linked to Streamlit Cloud, where the app was successfully deployed and made accessible through a permanent public URL.

---

## APP Link: https://insurancecostprediction-elqtq22jnkhtcd4ockjpt2.streamlit.app/


