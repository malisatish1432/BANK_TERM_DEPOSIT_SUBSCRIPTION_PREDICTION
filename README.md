# BANK_TERM_DEPOSIT_SUBSCRIPTION_PREDICTION
📌 Bank Term Deposit Subscription Prediction
🏦 Project Overview

This project predicts whether a customer will subscribe to a term deposit using machine learning.
It is based on a Portuguese bank's marketing campaign dataset. The goal is to improve marketing efficiency by identifying potential subscribers in advance.

🎯 Why It Matters

Reduce marketing costs by focusing only on high-potential clients

Improve campaign success rates with targeted offers

Enhance customer experience by avoiding irrelevant calls

📊 Dataset Summary

Target Variable: y (Binary: yes / no)
Features include:

Demographic: Age, Job, Marital Status, Education

Financial: Balance, Housing Loan, Personal Loan

Campaign-related: Contact Type, Last Contact Duration, Number of Contacts, Previous Outcome

Economic Indicators: Employment Rate, Consumer Confidence Index, Interest Rates

🛠 Approach

Data Preprocessing

Missing value handling

Encoding categorical variables

Feature scaling

Data balancing using SMOTE/SMOTENC

Model Training & Comparison

Logistic Regression

Decision Tree

Random Forest

XGBoost

LightGBM

CatBoost (best performer)

Gradient Boosting, KNN, Voting Classifier, AdaBoost

Evaluation Metrics

Accuracy

Precision, Recall, F1-score

ROC-AUC

Confusion Matrix

🚀 Results
Model	Accuracy	ROC-AUC
CatBoost	0.8917	0.9176
RandomForest	0.8750	0.9054
XGBoost	0.8700	0.9030

📌 CatBoost was selected as the final model for its high accuracy, excellent AUC score, and low need for hyperparameter tuning.

📈 Visualizations
ROC Curve

Confusion Matrix

Feature Importance (CatBoost)

🖥 Tech Stack

Python: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

ML Libraries: XGBoost, LightGBM, CatBoost, Imbalanced-learn

Environment: Jupyter Notebook

📜 Conclusion

Using CatBoost, we achieved 91.76% ROC-AUC and 89.17% accuracy, successfully identifying likely term deposit subscribers. This solution can help banks save resources and increase campaign conversion rates.
