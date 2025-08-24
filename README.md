# 💰 Bank Term Deposit Subscription Prediction

**Predicting customer subscription to term deposits using machine learning classification models**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)]()
[![RandomForest](https://img.shields.io/badge/RandomForest-Classifier-green)]()
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
- AdaBoost Classifier  
- KNN  
- SVM  
- XGBoost Classifier 

---

## 📊 Model Performance

| Model                | Accuracy | Precision | Recall  | F1-score | ROC-AUC |
|----------------------|----------|-----------|---------|----------|---------|
| Logistic Regression  | 0.8939   | 0.5294    | 0.1837  | 0.2727   | 0.8644  |
| Decision Tree        | 0.8729   | 0.4220    | 0.4694  | 0.4444   | 0.6957  |
| **Random Forest Classifier ✅**   | **0.8972** | **0.5641** | **0.2245** | **0.3212** | **0.9120** |
| Gradient Boosting Classifier    | 0.8928   | 0.5094    | 0.2755  | 0.3576   | 0.9019  |
| AdaBoost Classifier             | 0.8994   | 0.5538    | 0.3673  | 0.4417   | 0.8905  |
| KNN                  | 0.8917   | 0.5000    | 0.2041  | 0.2899   | 0.7393  |
| SVM                  | 0.8961   | 0.5556    | 0.2041  | 0.2985   | 0.8245  |
| XGBoost Classifier              | 0.8972   | 0.5352    | 0.3878  | 0.4497   | 0.9052  |

**🏆 Best Model:** **Random Forest Classifier**  
**🔥 Key Metrics:** Accuracy = 0.8972, Precision = 0.5641, Recall = 0.2245, F1-score = 0.3212, ROC-AUC = 0.9120

> ⚠️ *Note:* `duration` is a strong predictor but may be **leaky** if used for pre-call targeting. Consider deploying a variant **without** it for production.

---

## 📈 Visualizations
- **ROC Curve**  
- **Confusion Matrix**  
- **Feature Importance Plot**

---

## 📌 Insights
- 📞 **Call duration** strongly affects subscription probability  
- ✅ Positive **previous campaign outcomes** increase conversion  
- 💰 Higher **balances** and certain **jobs** correlate with better subscription rates

---

## ⚠️ Limitations
- Class imbalance (more “no” than “yes”) reduces **recall**  
- Campaign-specific features (e.g., `duration`) may not be available pre-contact  
- Socio-economic features may **drift** over time

---

## 🚀 Future Work
- Apply **SMOTE** or class-weighting to handle imbalance  
- Explore **stacked ensembles** or **probability calibration**  
- Build a real-time scoring **Streamlit/Flask** app  
- Add **drift monitoring** and scheduled **retraining**

---

## 🛠 Tools & Libraries  
- Python 🐍  
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  
- RandomForest, XGBoost  

---

## 📜 Conclusion  
This project demonstrates how **Random Forest Classifier** can effectively predict bank term deposit subscriptions.  
With preprocessing, feature engineering, and hyperparameter tuning, we achieved **Accuracy = 0.8972** and **ROC-AUC = 0.9120**, supporting better marketing targeting and decision-making.

---

## 👤 Author  
**Mali Satish**  
🎓 Machine Learning Enthusiast | Data Science Student @ BHU





 

