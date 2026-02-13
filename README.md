# Credit card transactions - Fraud detection: Evaluating machine learning models and determining abnormal features
<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/3f92281c-9711-4f53-b25a-b6a1ee548d3e" />

By: Jessica Tran - me 
# Project summary
This project focuses on detecting fraudulent transactions using machine learning models. Due to class imbalance in the dataset, traditional evaluation metrics are insufficient. The project emphasises fraud-specific metrics and model interpretability to identify the most effective predictive approach.
This project aims to:
  1. Analyse patterns in anonymised credit card transaction data and identify characteristics associated with fraudulent behaviour
  2. To develop and compare ensemble models for fraud detection
  3. To evaluate machine learning models emphasizing recall and F1-score as primary metrics for imbalanced data
  4. To determine the most suitable model for minimising missed fraudulent transactions

Tools used: Python (Pandas, Seaborn, Matplotlib, Scikit-learn)
# Dataset description
Data source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The real-world bank transactions by European cardholders in 2013 are presented is this data. Out of more than 284K transactions, 492 were recorded as fraud and took place over than 2 days.
There are 31 columns, including:
 - Time: by seconds, since the very first transaction was occured
 - V1 - V28: feature columns - results of PCA transformation
 - Amount: the amount that each transaction processes
 - Class: fraud - normal category (0: normal; 1:fraud)
# Explotary data analysis (EDA)
## Class imbalance
<img width="1536" height="754" alt="Image" src="https://github.com/user-attachments/assets/ddb8d6d5-3ae1-4bfa-91e6-cc674315d643" />

The dataset is extremely imbalanced, with the number of fraudulent transactions accounting for 0.17% of all transactions. Therefore, accuracy is not recommended in this analysis to evaluate the models. Instead, suitable metrics, such as Recall, Precision, F1-score, AUPRC are highlighted for this imbalance classification

## Transaction amount analysis
<img width="600" height="400" alt="Image" src="https://github.com/user-attachments/assets/a7e108c1-5465-42cf-bbac-fad7ac080e7b" />

Overall, The transaction amount distribution is highly right-skewed, with a small number of transactions having very large values.

<img width="1200" height="600" alt="Image" src="https://github.com/user-attachments/assets/df788bd8-1de5-45cb-b8fd-40cff220f24d" />

The boxplot shows that both normal and fraudulent transactions exhibit a right-skewed distribution of transaction amounts. Normal transactions has higher median and more outliers due to the much large number of normal cases, while lower median and high density of outliers are observed in type fraud because of small number of anonymised records. This huge overlap indicates that the amount feature alone is not a strong predictor for fraud detection and it may lead to missed fraudulent cases as many fraud transactions fall within the typical amount range of normal transactions.

## Time-based analysis
<img width="1000" height="600" alt="Image" src="https://github.com/user-attachments/assets/23cb0b7f-5306-480e-8f2d-031658ff603c" />

Transaction time was converted into hourly bins to examine temporal patterns. The variations across time hour are observed in the distribution of fraud amount and the density of amount for both transactions are mostly overlapped, showing that time-related features provide limited potential compared to behavioural features.

<img width="1000" height="600" alt="Image" src="https://github.com/user-attachments/assets/e9c820ec-1a44-42a4-959a-98e050464691" />

The fraudulent rate over time is unstable with the number of peaks occuring incoherently. This can be explained by low transaction volumes in these time hours. There is no particular time patterns to predict fraudulent behaviours, meaning that transaction time alone exhibit insufficient potential for fraud detection.

Consequently, compared to transaction amount and time, behavioural PCA features analysis is required to determine clearer separation between fraudulent and legitimate transactions.

## Feature correlation
<img width="1536" height="754" alt="Image" src="https://github.com/user-attachments/assets/9bd79836-616f-45ed-8569-7f2175ae82b8" />

Correlation heat map shows the limited association between most features. This is expected as the majority of predictors are PCA-transformed components.

# Data preprocessing
- Duplicate records were removed
- Transaction amount was standardalized using StandardScaler()
- Stratified train–test splitting was applied to preserve the original class distribution

# Model development
Four supervised learning models were implemented:
  1. Logistic Regression: baseline model for transaction type classification, used to assess whether linear decision boundaries are sufficient for fraud detection.
  2. Decision Tree: simple tree-based benchmark to capture non-linear relationships between characteristics
  3. Random Forest: more complex model to improve robustness through ensemble learning and reduce overfitting
  4. XGBoost Classifier: the star of this project to handle complex feature interaction in highly imbalanced classification tasks
# Model evaluation 
<img width="601" height="115" alt="Image" src="https://github.com/user-attachments/assets/f583fb63-7888-4faf-8962-b11a45581041" />

<img width="1000" height="600" alt="Image" src="https://github.com/user-attachments/assets/8e7694ad-398b-423a-b713-4043969e1c3a" />

As expected, accuracy scores are nearly 100 in all models, showing that accuracy metric is overfiiting to evaluate the best model in this case. XGBoost Classifier achieve the highest precision and F1 score, indicating the best balance and accurate model for fraud detection. Although Random Forest has the higher recall score, its precision is the lowest number. As a result, XGBoost provides a superior balance between precision and recall

# Confusion matrix
<img width="400" height="400" alt="Image" src="https://github.com/user-attachments/assets/66637933-7058-4c82-a156-53f82ed5b9e5" />
<img width="400" height="400" alt="Image" src="https://github.com/user-attachments/assets/d6f6cb7a-01ca-4f8b-a6b4-1ecc3f5d9243" />

Although Random Forest detects one more fraudulent transaction compared to XGBoost, it flagged more than 10 normal cases as fraud, causing higher costs to investigate. XGBoost Classifier, even missed 1 fraudulent transaction, significantly reduces false positives while maintaining comparable recall. Therefore, XGBoost provides a better overall balance

# Feature importance
## Random Forest 
<img width="800" height="600" alt="Image" src="https://github.com/user-attachments/assets/bbee1ee3-87db-43c0-954e-33551f7fe8a1" />

## XGBoost Classifier
<img width="800" height="600" alt="Image" src="https://github.com/user-attachments/assets/e545402b-cbe9-40f6-a92d-315bf077205f" />

Both charts of feature importance in Random Forest and XGBoost reveal the significance of feature V14, and certain behavioural features that contribute the most to fraud detection, including V4, V10, V12,... In Random Forest, it depends on multiple characteristics to evaluate and label and the contributions of several features are all considerable. Conversely, V14 is the most dominant feature in XGBoost Classifier, highlighting the reliable and predictive potential of feature V14 and the ability of XGBoost model to optimise information gain. Especially, transaction time and amount are not in top feature importance, indicating their ambiguous characteristics in fraud detection

# Distribution of top predictors
<img width="1200" height="600" alt="Image" src="https://github.com/user-attachments/assets/0647c76b-a119-454e-89f9-4cc42baf9ccf" />

Based on the feature importance analysis, the most influential features, including 'V14','V10','V4','V12','V11','V9', are selected for further distribution analysis. The Kernel Density Estimation (KDE) plots are used to compare thier significance. The plot show clear distribution shifts between the 2 classes, fraud and normal. Fraudulent cases generally exhibit sharper and more concetrated peaks, whereas normal ones are broader and more disperesed showing that these PCA-transformed predictors offer strong discriminatory power for fraud detection.
V14 and V10 demonstrate clear separation between fraud and normal cases, with smaller overlap and higher density peaks. Similarly, V4, V12, V11 have larger normal distribution and smaller peaks, highlighting thier predictive relevance. In contrast, V9 shows relatively greater overlap between fraud and normal and closer peak alignment, suggesting its weaker discriminative capability among these selected features.
Overall, the observed separation patterns confirm that these behavioural features are important to fraud detection, far more than transaction amount and time

# Conclusion
This project demonstrates an end-to-end fraud detection workflow, from exploratory data analysis to model selection and interpretation. XGBoost Classifier is the best model for fraud detection to detect fraud and optimise the costs of false negatives, evaluated by F1-score and Recall metrics. The results highlight the importance of machine learning models and behavioural features in handling highly imbalanced financial datasets

# Reference
1. https://towardsdatascience.com/credit-card-fraud-detection-using-machine-learning-python-5b098d4a8edc/
2. https://www.kaggle.com/code/gpreda/credit-card-fraud-detection-predictive-models
