# Credit card transactions - Fraud Detection: Optimising fraudulent rate and determining abnormal features
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/3f92281c-9711-4f53-b25a-b6a1ee548d3e" />

By: Jessica Tran - me 
## Project summary
This project focuses on detecting fraudulent transactions using machine learning models. Due to class imbalance in the dataset, traditional evaluation metrics are insufficient. The project emphasises fraud-specific metrics and model interpretability to identify the most effective predictive approach.
This project aims to:
  1. Analyse patterns in anonymised credit card transaction data and identify characteristics associated with fraudulent behaviour
  2. To develop and compare models for fraud detection
  3. To evaluate machine learning models emphasizing recall and F1-score as primary metrics for imbalanced data
  4. To determine the most suitable model for minimising missed fraudulent transactions
Tools used: Python (Pandas, Numpy, Seaborn, Matplotlib, Scikit-learn)
## Dataset description
The real-world bank transactions that are recored by European cardholders conducted in 2013 are presented is this data. Out of more than 284K transactions, 492 were recorded as fraud and took place over than 2 days.
There are 31 columns, including:
 - Time: by seconds, since the very first transaction was occured
 - V1 - V28 (29 feature columns - location, branch, transaction type,...): PCA transformation
 - Amount: the amount of money that each transaction processes
 - Class: fraud - normal category (0: normal; 1:fraud)
## Explotary data analysis (EDA)
### Class imbalance
<img width="1536" height="754" alt="Image" src="https://github.com/user-attachments/assets/ddb8d6d5-3ae1-4bfa-91e6-cc674315d643" />
The dataset is extremely imbalanced, with the number of fraudulent transactions accounting for 0.17% of above 284K records. Therefore, accuracy is not recommended in this analysis to evaluate the models. Instead, suitable metrics, such as Recall, Precision, F1-score, AUPRC are highlighted for this imbalance classification

### Amount distribution
<img width="600" height="400" alt="Image" src="https://github.com/user-attachments/assets/a7e108c1-5465-42cf-bbac-fad7ac080e7b" />
