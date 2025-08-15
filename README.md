# 💰 Bank Term Deposit Subscription Prediction

*Predicting customer subscription to term deposits using machine learning classification models*

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
- [Insights](#-insights)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [How to Run](#-how-to-run)
- [Tools & Libraries](#-tools--libraries)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

## 🔍 Overview
This project predicts whether a **bank customer** will subscribe to a **term deposit** based on demographic, social, and economic attributes. Accurate predictions help banks:
- Target **potential subscribers** effectively  
- Reduce **marketing costs**  
- Improve **conversion rates**  
- Personalize **customer engagement** strategies

---

## 🎯 Problem Statement
- **Goal:** Predict whether a customer will subscribe to a term deposit (`yes` or `no`)  
- **Type:** **Binary Classification**  
- **Target Variable:** `y` *(subscription status)*

---

## 📁 Dataset
- **Name:** Bank Marketing Dataset  
- **Source:** UCI Machine Learning Repository / Kaggle  
- **Shape:** `(45211, 17)`

**Features**

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Age of the client |
| job | Categorical | Job type |
| marital | Categorical | Marital status |
| education | Categorical | Education level |
| default | Binary | Credit in default? |
| balance | Numeric | Avg yearly balance (EUR) |
| housing | Binary | Housing loan status |
| loan | Binary | Personal loan status |
| contact | Categorical | Contact communication type |
| day | Numeric | Last contact day of the month |
| month | Categorical | Last contact month of year |
| duration | Numeric | Last contact duration (seconds) |
| campaign | Numeric | Contacts during current campaign |
| pdays | Numeric | Days since last contact (−1 = never) |
| previous | Numeric | Contacts before this campaign |
| poutcome | Categorical | Outcome of previous campaign |
| **y (Target)** | Binary | Subscribed? (`yes`/`no`) |

---

## 🔄 Workflow

```mermaid
flowchart LR
    A[Raw Data] --> B[Data Cleaning<br/>missing values, IQR outliers]
    B --> C[Encoding & Scaling<br/>OHE/LabelEnc + StandardScaler]
    C --> D[Train/Test Split (80/20)]
    D --> E[Model Training<br/>LR / DT / RF / GB / CatBoost]
    E --> F[Evaluation<br/>Accuracy • Precision • Recall • F1 • ROC-AUC]
    F --> G[Best Model Selection<br/>CatBoost]
    G --> H[Explainability<br/>Feature Importance, Confusion Matrix, ROC]

