# Credit card transactions - Fraud detection: Evaluating machine learning models and determining abnormal features
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/3f92281c-9711-4f53-b25a-b6a1ee548d3e" />

By: Jessica Tran - me 
# 📖 Project summary
This project focuses on detecting fraudulent transactions using machine learning models. Due to class imbalance in the dataset, traditional evaluation metrics are insufficient. The project emphasises fraud-specific metrics and model interpretability to identify the most effective predictive approach.
This project aims to:
  1. Analyse patterns in anonymised credit card transaction data and identify characteristics associated with fraudulent behaviour
  2. To develop and compare ensemble models for fraud detection
  3. To evaluate machine learning models emphasizing recall and F1-score as primary metrics for imbalanced data
  4. To determine the most suitable model for minimising missed fraudulent transactions

Tools used: Python (Pandas, Seaborn, Matplotlib, Scikit-learn)
# 💾 Dataset description
Data source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The real-world bank transactions by European cardholders in 2013 are presented is this data. Out of more than 284K transactions, 492 were recorded as fraud and took place over than 2 days.
There are 31 columns, including:
 - Time: by seconds, since the very first transaction was occured
 - V1 - V28: feature columns - results of PCA transformation
 - Amount: the amount that each transaction processes
 - Class: fraud - normal category (0: normal; 1:fraud)

# Methodologies

1. Data loading and understanding
2. Data cleaning
- Conducted data cleaning by removing duplicate transaction records and validating dataset quality
- Performed preprocessing using feature scaling (StandardScaler) and stratified train-test splitting to preserve fraud distribution.
3. Exploratory Data Analysis (EDA)
- Applied exploratory data analysis (EDA) to examine class imbalance, transaction behaviour, amount distribution, time patterns, and feature correlations.
4. Feature engineering and preprocessing
5. Train-test splitting
6. Machine learning model development
  - Linear Regression
  - Decision Tree
  - Random Forests
  - XGBoost Classifier
7. Model evaluation and comparison
- Evaluated model performance using fraud-focused metrics including Recall, Precision, F1-score, and Confusion Matrix analysis.
8. Feature importance interpretation
- Conducted feature importance and distribution analysis to identify abnormal behavioural predictors associated with fraudulent transactions.
9. Business insights and recommendations
- Interpreted model performance from a business perspective by considering the trade-off between false positives and false negatives.

# Key findings 
- Fraud transactions are extremely rare, creating severe class imbalance.
- Ensemble learning models outperform simpler baseline methods.
- XGBoost provides the strongest fraud detection performance.
- Certain anonymized variables contribute heavily to fraud classification.
- Evaluation focusing on Recall and Precision is essential for fraud detection systems

# Recommendations
- Deploy ensemble-based fraud detection models such as XGBoost in production systems
- Prioritize recall to minimize undetected fraud cases
- Implement real-time fraud monitoring during high-risk transaction periods
- Continuously retrain models to adapt to evolving fraud behavior
- Balance the trade-off between false positives and false negatives to optimize investigation costs and fraud prevention effectiveness.
- Combine machine learning predictions with rule-based monitoring systems for improved reliability

# 📌 Conclusion
This project demonstrates an end-to-end fraud detection workflow, from exploratory data analysis to model selection and interpretation. XGBoost Classifier is the best model for fraud detection to detect fraud and optimise the costs of false negatives, evaluated by F1-score and Recall metrics. The results highlight the importance of machine learning models and behavioural features in handling highly imbalanced financial datasets

# Reference
1. https://towardsdatascience.com/credit-card-fraud-detection-using-machine-learning-python-5b098d4a8edc/
2. https://www.kaggle.com/code/gpreda/credit-card-fraud-detection-predictive-models
