
 # 📌 Bank Term Deposit Subscription Prediction : 
 
## 🏦 Project Overview :
### 🔹 This project predicts whether a customer will subscribe to a term deposit using machine learning.It is based on a Portuguese bank's marketing campaign dataset. The goal is to improve marketing efficiency by identifying potential  subscribers in advance.
## 🎯 Why It Matters :
### 🔹 Reduce marketing costs by focusing only on high-potential clients.
### 🔹 Improve campaign success rates with targeted offers.
### 🔹 Enhance customer experience by avoiding irrelevant calls.
## 📊 Dataset Summary :
### 🔹Target Variable: y (Binary: yes / no)
### 🔹Features include:
   ### 🔹Demographic: Age, Job, Marital Status, Education .
   ### 🔹Financial: Balance, Housing Loan, Personal Loan .
   ### 🔹Campaign-related: Contact Type, Last Contact Duration, Number of Contacts, Previous Outcome .
   ### 🔹Economic Indicators: Employment Rate, Consumer Confidence Index, Interest Rates .
## 🛠 Approach :
 ### 🔹Data Preprocessing
 ### 🔹Missing value handling
 ### 🔹Encoding categorical variables
 ### 🔹Feature scaling
 ### 🔹Data balancing using SMOTE/SMOTENC
 ### 🔹Model Training & Comparison
 ### 🔹Logistic Regression
 ### 🔹Decision Tree
 ### 🔹Random Forest
 ### 🔹XGBoost
 ### 🔹LightGBM
 ### 🔹CatBoost (best performer)
 ### 🔹Gradient Boosting, KNN, Voting Classifier, AdaBoost
 ### 🔹Evaluation Metrics
 ### 🔹Accuracy
	### 🔹Precision, Recall, F1-score
 ### 🔹ROC-AUC
	### 🔹Confusion Matrix
## 🚀 Results :
## 📊 Classification Model Performance Summary  

| Model                 | Accuracy  | Precision | Recall  | F1-score | ROC-AUC |
|-----------------------|-----------|-----------|---------|----------|---------|
| CatBoost              | 0.8976    | 0.5738    | 0.3571  | 0.4403   | 0.9176  |
| Logistic Regression   | 0.8939    | 0.5294    | 0.1837  | 0.2727   | 0.8644  |
| Decision Tree         | 0.8729    | 0.4220    | 0.4694  | 0.4444   | 0.6957  |
| Random Forest         | 0.8928.   | 0.6087    | 0.2857  | 0.3889   | 0.9120  |
| Gradient Boosting     | 0.8928    | 0.5094    | 0.2755  | 0.3576   | 0.8019  |
| AdaBoost              | 0.8994    | 0.5538    | 0.3673  | 0.4417   | 0.8905  |
| KNN                   | 0.8917    | 0.5000    | 0.2041  | 0.2899   | 0.7393  |
| SVM                   | 0.8961    | 0.5556    | 0.2041  | 0.2985   | 0.8245  |
| LightGBM              | 0.8950    | 0.5211    | 0.3776  | 0.4379   | 0.9035  |
| XGBoost               | 0.8972    | 0.5352    | 0.3878  | 0.4497   | 0.9052  |

 ### 📌 CatBoost was selected as the final model for its high accuracy, excellent AUC score, and low need for hyperparameter tuning.
## 📈 Visualizations
### 🔹 ROC Curve
### 🔹 Confusion Matrix
### 🔹 Feature Importance (CatBoost)
## 🖥 Tech Stack :
  ### 🔹Python: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
  ### 🔹ML Libraries: XGBoost, LightGBM, CatBoost, Imbalanced-learn
  ### 🔹Environment: Jupyter Notebook
## 📜 Conclusion
 ### 🔹Using CatBoost, we achieved 91.76% ROC-AUC and 89.76% accuracy, successfully identifying likely term deposit subscribers. This solution can help banks save resources and increase campaign conversion rates.
