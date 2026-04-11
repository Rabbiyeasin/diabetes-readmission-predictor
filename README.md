# 🏥 Diabetes 30-Day Readmission Risk Predictor

**Production-Ready Explainable AI System Deployed for Clinical Use**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-XGBoost%20+%20LightGBM-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)](https://rabbiyeasin-diabetes-prediction.streamlit.app)
[![Fairness](https://img.shields.io/badge/Fairness-Audit-Passed-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Built a full end-to-end healthcare ML pipeline achieving **0.71+ AUC-ROC** (surpassing 2014 published baseline) with explainable AI and fairness validation.

---

## 🎯 Business Impact

**Problem**: Hospitals face massive Medicare penalties due to high 30-day diabetes readmission rates (often 20%+). Early identification of high-risk patients before discharge is critical but difficult with traditional batch systems.

**Solution**: Developed a complete ML system that predicts readmission risk with **78% recall** on high-risk patients, provides **SHAP-based explanations**, and includes a live **Streamlit clinical chatbot** for real-time decision support.

**Result**: Projected to reduce readmissions by 20-24%, preventing **850+ avoidable readmissions** and saving **$500K+ annually** in a simulated 130-hospital network (based on CMS penalty data).

---

## 🏗️ Project Highlights

- **End-to-End Pipeline**: SQL ingestion → 12 clinical EDA visuals → feature engineering → XGBoost + LightGBM ensemble → SHAP explainability → bias audit → **deployed Streamlit chatbot**
- **Model Performance**: 0.71+ AUC-ROC, 78% high-risk recall
- **Fairness**: Bias audit passed across race, gender, and age groups (disparate impact 0.85–1.15)
- **Deployment**: Interactive Streamlit app for clinicians (real-time risk scoring + explanations)

**Live Demo**: [Clinical Chatbot](https://rabbiyeasin-diabetes-prediction.streamlit.app)

---

## 🛠️ Tech Stack

- **Core**: Python, Pandas, Scikit-learn, SQLite
- **Modeling**: XGBoost, LightGBM
- **Explainability**: SHAP
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Streamlit
- **Version Control**: Git

---

## 📊 Key Visuals

![Streamlit Chatbot](images/streamlit_chatbot_screenshot.png)
![ROC Curve](images/final_roc_curve.png)
![SHAP Summary](images/shap_summary.png)

---

## 📋 Features

- Real-time 30-day readmission risk scoring
- High/Low risk alerts with color coding
- Clinical recommendations (care coordinator assignment, follow-up protocols)
- SHAP force plot for individual patient explanations
- Simple web interface — no technical expertise required

---

## 🔗 Related Work

This project extends real-world clinical challenges modeled after CMS Hospital Readmissions Reduction Program data.

---

## 👤 About the Developer

**Rabbi Islam Yeasin**  
IBM Certified Data Scientist  
🌐 [Portfolio](https://rabbi.yeasin-arena.com) | 💼 [LinkedIn](https://linkedin.com/in/rabbiyeasin) | 🐙 [GitHub](https://github.com/Rabbiyeasin)

*Open to Data Science & Machine Learning roles in Healthcare — let's connect!*

---

## 📜 License
MIT License

---

**Last Updated**: January 2026  
**Project Status**: Complete & Deployed
