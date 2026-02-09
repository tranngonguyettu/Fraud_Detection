# Credit card transactions - Fraud Detection: Optimising fraudulent rate and determining abnormal features
Tools used: Python (Pandas, Numpy, Seaborn, Matplotlib, Scikit-learn)
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/3f92281c-9711-4f53-b25a-b6a1ee548d3e" />
By: Jessica Tran - me 
## Project description
This project focuses on detecting fraudulent transactions using machine learning models. Due to class imbalance in the dataset, traditional evaluation metrics are insufficient. The project emphasises fraud-specific metrics and model interpretability to identify the most effective predictive approach.
This project aims to:
  1. Analyse patterns in anonymised credit card transaction data and identify characteristics associated with fraudulent behaviour
  2. To develop and compare models for fraud detection
  3. To evaluate machine learning models using recall and F1-score as primary metrics for imbalanced data
  4. To determine the most suitable model for minimising missed fraudulent transactions

## Dataset description
The real-world bank transactions that are recored by European cardholders conducted in 2013 are presented is this data. Out of more than 284K transactions, 492 were recorded as fraud and took place over than 2 days.
There are 31 columns, including:
 - Time: by seconds, since the very first transaction was occured
 - V1 - V28 (29 feature columns - location, branch, transaction type,...): PCA transformation
 - Amount: the amount of money that each transaction processes
 - Class: fraud - normal category (0: normal; 1:fraud)
