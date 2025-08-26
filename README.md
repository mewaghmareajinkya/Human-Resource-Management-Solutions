"Human Resource Management Solutions"

This repository provides a Human Resource Analytics and Prediction System that uses machine learning to analyze employee data and predict attrition (whether an employee is likely to leave or stay). The project follows a streamlined data science workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and hyperparameter tuning.



📌 Project Overview

Employee retention is a critical challenge for organizations. High employee turnover can result in financial loss, operational inefficiency, and reduced morale among the workforce.

This project:

Cleans and preprocesses employee data.

Performs EDA to understand key factors influencing employee decisions.

Builds and evaluates a Random Forest Classifier for predicting employee attrition.

Optimizes the model using cross-validation and hyperparameter tuning.



⚙️ Technologies Used
Python

Libraries:

numpy, pandas – Data manipulation and preprocessing

matplotlib, seaborn – Data visualization

scikit-learn – Machine learning (modeling, evaluation, hyperparameter tuning)



📂 Dataset Details
File: Employee_Dataset.csv

Shape: (14999, 10) records before cleaning

After removing duplicates: (11991, 10) records

Features:

satisfaction_level (float)

last_evaluation (float)

number_project (int)

average_montly_hours (int)

time_spend_company (int)

Work_accident (int)

promotion_last_5years (int)

Department (categorical)

salary (categorical)

left (target – 1 = employee left, 0 = stayed)



📊 Exploratory Data Analysis
Key insights from data analysis:

Salary vs Retention: Employees with higher salaries are less likely to leave.

Department vs Retention: Attrition varies across departments, with sales and support showing higher turnover.

Satisfaction Level: Low satisfaction strongly correlates with attrition.

Work Tenure: Longer tenure employees show higher likelihood of leaving.

Visualizations included:

Bar charts (salary vs retention, department vs retention).

Distribution plots (satisfaction, project count, evaluation scores).

Boxplots of numerical features.



🛠️ Feature Engineering
Encoding categorical variables: Department and salary encoded using LabelEncoder.

Feature scaling: StandardScaler applied for normalization.

Splitting data: Train-test split (80-20).



🤖 Model Development
Model Used: Random Forest Classifier

Steps:

Data preprocessing and scaling

Random Forest training on scaled dataset

K-Fold Cross Validation (cv=5)

Hyperparameter tuning using GridSearchCV



📈 Model Evaluation
Confusion Matrix:

text
[[1991    7]
 [  39  362]]
Accuracy: 98.08%

Precision: 98.1%

Recall: 90.2%

F1-Score: 94.0%

Cross Validation Score: ~98.5%

This shows strong predictive power with high generalization across folds.



🔎 Feature Importance
Top contributing factors for employee attrition:

Satisfaction level

Number of projects

Time spent at company

Average monthly hours

Last evaluation



🔧 Hyperparameter Tuning
Parameters tested:

n_estimators: [50, 100]

max_features: ['sqrt', 'log2', None]

Best parameters found:

text
{'max_features': 'log2', 'n_estimators': 50}
New Model Average Accuracy: 98.55%



🚀 How to Run the Project
Clone the repo:

bash
git clone //github.com/mewaghmareajinkya/Human-Resource-Management-Solutions.git
cd Human-Resource-Management-Solutions

Install dependencies:
bash
pip install -r requirements.txt
Place the dataset (Employee_Dataset.csv) in the project root.

Run the Jupyter Notebook or Python scripts to reproduce the analysis:

bash
jupyter notebook HR_Analysis.ipynb



✅ Results
Built a highly accurate attrition prediction model.

Identified satisfaction level, work tenure, and number of projects as the most critical features.

Provided insights to HR for reducing turnover and improving retention strategies.



📌 Future Enhancements
Deploy model as a Flask/Django web app for HR teams.

Add more ML models (Logistic Regression, XGBoost, Neural Networks) for comparison.

Enhance interpretability with SHAP or LIME explainability methods.


