# 💰 Bank Term Deposit Subscription Prediction

**Predicting customer subscription to term deposits using machine learning classification models**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)]()
[![CatBoost](https://img.shields.io/badge/CatBoost-Classifier-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 🧭 Table of Contents
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Model Performance](#-model-performance)
- [Visualizations](#-visualizations)
- [Insights](#-insights)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Tools & Libraries](#-tools--libraries)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

## 🔍 Overview
This project predicts whether a **bank customer** will subscribe to a **term deposit** based on demographic, social, and economic attributes.

Accurate predictions help banks:
- 🎯 Target potential subscribers effectively  
- 💸 Reduce marketing costs  
- 📈 Improve conversion rates  
- 🤝 Personalize customer engagement strategies

---

## 📊 Problem Statement
- **Goal:** Predict whether a customer will subscribe to a term deposit (yes/no)  
- **Type:** Binary Classification Problem  
- **Target Variable:** `y` (subscription status)

---

## 📁 Dataset
- **Name:** Bank Marketing Dataset  
- **Source:** UCI Machine Learning Repository / Kaggle  
- **Shape:** `(45211, 17)`

| Feature     | Type         | Description |
|-------------|--------------|-------------|
| age         | Numeric      | Age of the client |
| job         | Categorical  | Job type |
| marital     | Categorical  | Marital status |
| education   | Categorical  | Education level |
| default     | Binary       | Credit in default? |
| balance     | Numeric      | Average yearly balance (in euros) |
| housing     | Binary       | Housing loan status |
| loan        | Binary       | Personal loan status |
| contact     | Categorical  | Contact communication type |
| day         | Numeric      | Last contact day of the month |
| month       | Categorical  | Last contact month of year |
| duration    | Numeric      | Last contact duration (in seconds) |
| campaign    | Numeric      | Number of contacts during campaign |
| pdays       | Numeric      | Days since last contact (-1 = never) |
| previous    | Numeric      | Number of contacts before this campaign |
| poutcome    | Categorical  | Outcome of previous marketing campaign |
| **y (Target)** | Binary    | Has the client subscribed? (yes/no) |

---

## 🧪 Workflow

1️⃣ **Data Understanding**  
- Loaded dataset using pandas  
- Checked missing values, duplicates, and data types

2️⃣ **Data Cleaning & Preprocessing**  
- Converted categorical variables to numeric (Label Encoding / One-Hot Encoding)  
- Handled missing values  
- Outlier detection and removal (IQR method)  
- Standardized numerical features using `StandardScaler`

3️⃣ **Exploratory Data Analysis (EDA)**  
- Histograms & boxplots for numerical variables  
- Subscription rates vs. categorical features  
- Correlation heatmap  
- Target distribution analysis

4️⃣ **Feature Engineering**  
- Derived new features (e.g., age groups)  
- Removed irrelevant columns

5️⃣ **Data Splitting**  
- Train/Test split: **80% / 20%**

6️⃣ **Model Training**  
Evaluated multiple algorithms:
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- CatBoost Classifier

---

## 📊 Model Performance

| Model                | Accuracy | Precision | Recall  | F1-score | ROC-AUC |
|----------------------|----------|-----------|---------|----------|---------|
| Logistic Regression  | 0.8939   | 0.5294    | 0.1837  | 0.2727   | 0.8644  |
| Decision Tree        | 0.8729   | 0.4220    | 0.3571  | 0.3860   | 0.7981  |
| Random Forest        | 0.8965   | 0.5556    | 0.2449  | 0.3404   | 0.9127  |
| Gradient Boosting    | 0.8990   | 0.5517    | 0.2653  | 0.3579   | 0.9178  |
| **CatBoost ✅**       | **0.9017** | **0.5738** | **0.3571** | **0.4403** | **0.9233** |

**🏆 Best Model:** **CatBoost Classifier** (highest ROC-AUC = **0.9233**)  
**🔥 Key Predictors:** `duration`, `contact`, `poutcome`, `balance`, `age`

> ⚠️ *Note:* `duration` is a strong signal but can be **leaky** if used for pre-call targeting. Consider training a variant **without** `duration` for deployment.

---

## 📈 Visualizations
- **ROC Curve**  
- **Confusion Matrix**  
- **Feature Importance Plot**

---

## 📌 Insights
- 📞 **Call duration** is the strongest predictor — longer calls → higher chance of subscription  
- ✅ Positive **previous campaign outcomes** increase conversion likelihood  
- 💰 Higher **balances** and certain **jobs** (e.g., management, technician) correlate with higher subscription rates

---

## ⚠️ Limitations
- Class imbalance (more “no” than “yes”) reduces **recall**  
- Socio-economic features may **drift** over time  
- Campaign-specific features (e.g., `duration`) might not be available **pre-contact**

---

## 🚀 Future Work
- Apply **SMOTE** or class-weighting to handle imbalance  
- Explore **stacked ensembles** and **probability calibration**  
- Build a real-time scoring **Streamlit/Flask** app  
- Add **drift monitoring** and scheduled **retraining**

---

## 🛠 Tools & Libraries  
- Python 🐍  
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  
- CatBoost, XGBoost  

---

## 📜 Conclusion  
This project demonstrates how machine learning classification can help predict **bank term deposit subscriptions**.  
Using **CatBoost Classifier** with preprocessing, feature engineering, and tuning, we achieved **ROC-AUC = 0.9233**.  

Such models can help banks:  
- 📊 Improve marketing efficiency  
- 📈 Increase conversion rates  
- 💵 Reduce unnecessary campaign costs  

---

## 👤 Author  
**Mali Satish**  
🎓 Machine Learning Enthusiast | Data Science Student @ BHU  ---




 

