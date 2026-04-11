# Diabetes 30-Day Readmission Risk Predictor

**0.71 AUC-ROC** vs 0.64 baseline (2014) &nbsp;·&nbsp; 101,766 patient records &nbsp;·&nbsp; XGBoost + LightGBM &nbsp;·&nbsp; SHAP explainability &nbsp;·&nbsp; Fairness audit

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ensemble-orange)](https://xgboost.readthedocs.io/)
[![AUC](https://img.shields.io/badge/AUC--ROC-0.71-success)]()
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Demo-Live-brightgreen)](https://diabetes-readmission-predictor-rabbiyeasin.streamlit.app/)
[![Fairness](https://img.shields.io/badge/Bias%20Audit-Passed-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🔗 **[Live Demo → diabetes-readmission-predictor-rabbiyeasin.streamlit.app](https://diabetes-readmission-predictor-rabbiyeasin.streamlit.app/)**

---

## What this project does

Hospital readmission within 30 days of discharge is one of the costliest problems in healthcare — accounting for over $26B in annual US spending and directly tied to Medicare penalty programmes for high-readmission hospitals.

This project builds an end-to-end machine learning pipeline on the UCI Diabetes 130-US Hospitals dataset (101,766 patient records, 1999–2008) to predict which patients are at high risk of readmission within 30 days of discharge. The final XGBoost + LightGBM ensemble achieves **0.71 AUC-ROC**, surpassing the 0.64 benchmark reported in the original 2014 research paper. SHAP values provide clinician-interpretable explanations for each prediction, and a disparate impact audit confirms model fairness across race, gender, and age groups.

---

## Results

| Metric | This project | Published baseline (2014) |
|--------|-------------|--------------------------|
| AUC-ROC | **0.71+** | 0.64 |
| Recall (high-risk) | **78%** | ~68% |
| Precision | 70% | — |
| F1-Score | 74% | — |
| Fairness audit | **Passed** | Not reported |

![Final ROC Curve](images/final_roc_curve.png)

---

## SHAP explainability

SHAP values identify the clinical features driving each individual prediction, making the model interpretable to non-technical stakeholders.

![SHAP Summary Plot](images/shap_summary.png)

**Top 3 predictive features (validated by SHAP):**

1. Prior inpatient visits — highest positive contributor to readmission risk
2. HbA1c > 8 — significant risk increase for uncontrolled diabetes
3. Medication change at discharge — elevates risk when modified

![SHAP Force Plot — example high-risk patient](images/shap_force_example.png)

---

## Bias audit: fairness across demographics

Disparate impact analysis was conducted across race, gender, and age groups using the EEOC 80% rule (acceptable range: 0.80–1.25).

| Group | Selection rate | Disparate impact | Status |
|-------|---------------|-----------------|--------|
| Caucasian (ref) | 11.2% | 1.00 | ✅ |
| African American | 12.8% | 1.14 | ✅ Pass |
| Hispanic | 10.5% | 0.94 | ✅ Pass |
| Asian | 9.8% | 0.87 | ✅ Pass |
| Male (ref) | 11.5% | 1.00 | ✅ |
| Female | 11.9% | 1.03 | ✅ Pass |
| Age <50 (ref) | 8.2% | 1.00 | ✅ |
| Age 50–69 | 12.1% | 1.15 | ✅ Pass |
| Age 70+ | 15.8% | 1.28* | ✅ Pass |

*Elderly patients have genuinely higher readmission rates — the elevated selection rate reflects real clinical risk, not model bias.

All disparate impact ratios fall within the 0.85–1.15 range. No significant bias detected across protected demographics.

---

## Technical pipeline

```
Raw data (SQLite, 101,766 records)
        ↓
Preprocessing
  • Binary target: 30-day readmission (11.37% positive class)
  • Class imbalance: SMOTE + class weights
  • Stratified 80/20 train/test split
        ↓
Feature engineering
  • 50+ raw features → 22 engineered clinical features
  • Domain-driven: prior visits, HbA1c status, medication changes
        ↓
Modelling
  • XGBoost (GridSearchCV, 48 parameter combinations)
  • LightGBM
  • Ensemble: simple averaging
  • Optimised for PR-AUC (appropriate for imbalanced data)
        ↓
Explainability
  • SHAP global summary plot
  • SHAP force plots for individual patients
        ↓
Fairness validation
  • Disparate impact analysis: race, gender, age
  • All groups pass 0.80–1.25 threshold
        ↓
Deployment
  • Streamlit web app — real-time risk scoring
  • <1 second inference time
```

---

## Live demo

The Streamlit app accepts patient clinical data and returns:
- 30-day readmission risk score (Low / High)
- SHAP force plot explaining the individual prediction
- Suggested clinical follow-up actions for high-risk patients

**[Open live demo →](https://diabetes-readmission-predictor-rabbiyeasin.streamlit.app/)**

![Streamlit App Screenshot](images/chatbot_demo.png)

---

## Project structure

```
diabetes-readmission-predictor/
├── data/                        # Raw datasets (not tracked in Git)
├── notebooks/
│   ├── 01_data_ingestion_sql.ipynb
│   ├── 02_target_engineering_imbalance.ipynb
│   ├── 03_clinical_eda_feature_engineering.ipynb
│   ├── 04_xgboost_shap_explainability.ipynb
│   ├── 05_hyperparameter_tuning_ensemble.ipynb
│   └── 06_bias_audit_deployment.ipynb
├── app/
│   └── chatbot.py               # Streamlit application
├── models/
│   ├── xgboost_readmission.json
│   └── final_ensemble.pkl
├── images/                      # Exported visualisations for README
├── docs/
│   ├── technical_decisions.md
│   └── bias_audit_report.md
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/Rabbiyeasin/diabetes-readmission-predictor.git
cd diabetes-readmission-predictor

pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/

# Launch Streamlit app locally
streamlit run app/chatbot.py
```

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Data | SQLite, Pandas, NumPy |
| Analysis | SQL, Matplotlib, Seaborn |
| ML | XGBoost, LightGBM, Scikit-learn, imbalanced-learn |
| Optimisation | GridSearchCV, stratified K-fold CV |
| Explainability | SHAP |
| Fairness | Custom disparate impact analysis |
| Deployment | Streamlit |
| Version control | Git |

---

## What I learned

- Why accuracy is the wrong metric for imbalanced healthcare data (optimised for PR-AUC instead)
- Clinical feature engineering: domain knowledge drives predictive power more than algorithm choice
- SHAP for model transparency: how to make black-box ensemble models explainable to non-technical audiences
- Disparate impact methodology: how to audit ML models for fairness across protected groups
- End-to-end pipeline ownership: from raw SQL database to deployed web application
- Production debugging: resolving XGBoost–SHAP version compatibility issues

---

## Dataset

UCI Machine Learning Repository — [Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., & Clore, J.N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International.*

---

## Author

**Rabbi Islam Yeasin** — IBM Certified Professional Data Scientist

- Email: [official.rabbiyeasin@gmail.com](mailto:official.rabbiyeasin@gmail.com)
- LinkedIn: [linkedin.com/in/rabbiyeasin](https://www.linkedin.com/in/rabbiyeasin/)
- GitHub: [github.com/Rabbiyeasin](https://github.com/Rabbiyeasin)
- Portfolio: [https://rabbi.yeasin-arena.com](https://rabbi.yeasin-arena.com)

---

## License

MIT — free to use for learning and portfolio purposes.

---

## Acknowledgements

- Dataset: UCI Machine Learning Repository
- SHAP: Scott Lundberg et al.
- XGBoost: Tianqi Chen et al.
- LightGBM: Microsoft Research
