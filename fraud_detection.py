import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, classification_report, precision_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns 


data = pd.read_csv("creditcard.csv")

total_transactions = data.shape[0] #Total rows of transaction history
normal_transactions = data[data['Class']==0]    #Field of normal transactions
count_normal_trans = normal_transactions.shape[0] #number of normal  
fraud_transactions = data[data['Class']==1]     #field of fraud transactions
count_fraud_trans = fraud_transactions.shape[0]     #number of fraud

print("Total number of transactions:", total_transactions)
print("Number of normal transactions", count_normal_trans)
print("Number of fraud transactions:",count_fraud_trans)
print("Fraud transaction rate:", round((count_fraud_trans/total_transactions)*100,2)) #calculate the percentage of fraud: number of fraud trans/ total trans

#Glimpse the data
print("Credit card data - rows:", data.shape[0])
print("Credit card data - columns:", data.shape[1])
print(data.info())
print(data.head())
print(data.describe())
#No missing data


#Clean data
data.drop_duplicates(inplace = True) #drop duplicates
print(data.shape) #Check the data shape again 

#Data exploration
#Class distribution
class_counts = data['Class'].value_counts()

plt.figure(figsize=(6,4))
bars = plt.bar(class_counts.index, class_counts.values, width =0.2)
plt.xticks([0,1],['Normal','Fraud'])
plt.xlabel("Transaction type")
plt.ylabel("Number of transactions")
plt.title("Class distribution - normal vs fraud")

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval,
        f'{int(yval)}',
        ha='center',
        va='bottom'
    )    
plt.show()

#The number of fraud transaction is rare compared to the normal transactions
plt.figure(figsize=(6,4))
plt.hist(data["Amount"], bins = 100, color='red')
x_ticks = np.arange(min(data['Amount']), max(data['Amount']), 5000)
plt.xticks(x_ticks)
plt.xlabel('Transaction amount')
plt.ylabel("Frequency of amount")
plt.title("Amount distribution")
plt.tight_layout()
plt.show()
#The histogram is right-skewed, showing there is a huge difference in the amount values
#Imbalanced data may cause data bias

#Boxplots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

sns.boxplot(
    ax= ax1,
    x="Class",
    y= "Amount",
    hue="Class",
    data = data,
    palette="pastel",
    legend = False,
    showfliers = False
)
ax1.set_title("Amount by Class without outliers")
ax1.set_xlabel("Transaction classification")
ax1.set_ylabel("Transaction Amount")

sns.boxplot(
    ax= ax2,
    x="Class",
    y= "Amount",
    hue="Class",
    data = data,
    palette="pastel",
    showfliers = True,
    legend=False
)
ax2.set_title("Amount by Class with outliers")
ax2.set_xlabel("Transaction classification")
ax2.set_ylabel("Transaction Amount")
ax1.set_xticks([0,1])
ax1.set_xticklabels(["Normal", "Fraud"])
ax2.set_xticks([0,1])
ax2.set_xticklabels(["Normal", "Fraud"])
plt.tight_layout
plt.show()

#Time distribution: normal vs fraud
data['Hour'] = data['Time']/3600
print(data['Hour'])

normal_time = data[data['Class']==0]['Hour']
fraud_time = data[data['Class']==1]['Hour']

bins = np.linspace(data['Hour'].min(), data['Hour'].max(),50)
plt.figure(figsize=(10,6))
plt.hist(
    normal_time,
    bins = bins,
    density=True,
    alpha =0.3,
    label="Normal"
)
plt.hist(
    fraud_time,
    bins = bins,
    density=True,
    alpha=0.3,
    label="Fraud"
)
plt.xlabel("Time (Hour)")
plt.ylabel("Density")
plt.title("Transaction time distribution: Normal vs Fraud")
plt.legend()
plt.tight_layout()
plt.show()


#Fraud rate during time
data['Hourbin'] = data['Hour'].astype(int)
total_per_hour = data.groupby("Hourbin").size()
fraud_per_hour = data.groupby("Hourbin")['Class'].sum()
fraud_rate = (fraud_per_hour/total_per_hour)*100

plt.figure(figsize=(10,6))
plt.plot(fraud_rate.index, fraud_rate.values, color='red')
plt.xlabel("Time bin (hour)")
plt.ylabel("Fraud rate (%)")
plt.title("Fraud rate over hour")
plt.tight_layout()
plt.show()

#Feature correlation
plt.figure(figsize=(12,12))
corr = data.drop('Hourbin', axis = 1).corr()
sns.heatmap(corr, xticklabels=corr.columns, 
            yticklabels=corr.columns, linewidth=0.5,
            cmap='Blues')
plt.title("Feature correlation")
plt.show()


#Scaling is neccesary
sc = StandardScaler()
amount = data['Amount'].values
data['Amount'] = sc.fit_transform(amount.reshape(-1,1))


#Train and test split
X = data.drop('Class', axis = 1).values
y = data['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 42)

#Model building
#Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_prediction = log_reg.predict(X_test)
print("============================ Logistic Regression ============================")
print('Recall score of the Logistic Regression model is {}'.format(recall_score(y_test, log_reg_prediction)))
print('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, log_reg_prediction)))

#Decision tree
tree = DecisionTreeClassifier(criterion='gini',
                            max_depth=5,
                            random_state=42)
tree.fit(X_train, y_train)
tree_prediction = tree.predict(X_test)
print("============================ Decision Tree ============================")
print('Recall score of the Decision Tree model is {}'.format(recall_score(y_test, tree_prediction)))
print('F1 score of the Decision Tree model is {}'.format(f1_score(y_test, tree_prediction)))
confusion_matrix(y_test, tree_prediction, labels = [0, 1])

#Random Forest
forest = RandomForestClassifier(
    n_estimators=300,       #more trees, more stable
    max_depth = None,       #no limit to control overfitting by ensemble and bagging
    min_samples_leaf=30,    #decrease noise and smooth prediction
    class_weight='balanced',    #fraud focus
    n_jobs = -1,    
    random_state =42
)
forest.fit(X_train, y_train)
forest_prediction = forest.predict(X_test)
print("============================ Random Forest ============================")
print('Recall score of the Random Forest model is {}'.format(recall_score(y_test, forest_prediction)))
print('F1 score of the Random Forest model is {}'.format(f1_score(y_test, forest_prediction)))

#XGBoost classifier
neg = np.sum(y_train ==0)
pos = np.sum(y_train ==1)
scale_pos_weight = neg/pos
xgb = XGBClassifier(
    n_estimators = 300,
    max_depth = 5,
    learning_rate =0.5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    reg_lambda =1.0,
    scale_pos_weight = scale_pos_weight,
    objective ='binary:logistic',
    eval_metric ='logloss',
    n_jobs=-1,
    random_state =42
)

xgb.fit(X_train,y_train)
xgb_predict = xgb.predict(X_test)

print("============================ XGBoost Classifier ============================")
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_predict))
print("\nClassification Report:\n", classification_report(y_test, xgb_predict, digits=4))
print("Precision score of the Random Forest model is {}".format(precision_score(y_test, xgb_predict)))
print("Recall score of the Random Forest model is {}".format(recall_score(y_test, xgb_predict)))
print("F1-score of the XGBoost is {}".format(f1_score(y_test, xgb_predict)))

#Metric results table
results = []

models = {
    "Logistic Regression": log_reg_prediction,
    "Decision Tree": tree_prediction,
    "Random Forest": forest_prediction,
    "XGBoost Classifier": xgb_predict
}

for model_name, y_pred in models.items():
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 score": f1_score(y_test, y_pred)
    })

metrics_df = pd.DataFrame(results)
print(metrics_df)

#Eventually, XGBoost has the highest scores in Precision and F1
model = metrics_df['Model']
f1_score = metrics_df['F1 score']
recall_score = metrics_df['Recall']

x = np.arange(len(model))
width = 0.35

plt.figure(figsize=(10,6))
plt.bar(x - width/2, f1_score, width, label='F1 score', color = 'salmon')
plt.bar(x + width/2, recall_score, width, label='Recall', color = 'skyblue')

plt.xlabel("Model type")
plt.ylabel("score")
plt.title("F1 score and recall across models")
plt.xticks(x, models)
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Tính confusion matrix
cm1 = confusion_matrix(y_test, forest_prediction)
cm2 = confusion_matrix(y_test, xgb_predict)

# Tạo subplot
fig, axes = plt.subplots(1, 2, figsize=(10,4))

# Random Forest
sns.heatmap(
    cm1,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal','Fraud'],
    yticklabels=['Normal','Fraud'],
    ax=axes[0]
)
axes[0].set_title('Random Forest')
axes[0].set_xlabel('Predicted label')
axes[0].set_ylabel('True label')

# XGBoost
sns.heatmap(
    cm2,
    annot=True,
    fmt='d',
    cmap='Reds',
    xticklabels=['Normal','Fraud'],
    yticklabels=['Normal','Fraud'],
    ax=axes[1]
)
axes[1].set_title('XGBoost')
axes[1].set_xlabel('Predicted label')
axes[1].set_ylabel('True label')

plt.tight_layout()
plt.show()


#Feature importance
#1. Random Forest
feature_names = data.drop(columns=['Class']).columns
rf_importance = forest.feature_importances_
rf_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance':rf_importance
}).sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(8,6))
plt.barh(rf_imp_df['Feature'], rf_imp_df['Importance'], color = 'orange')
plt.gca().invert_yaxis()
plt.xlabel("Feature importance")
plt.title("Top 15 feature importance - Random Forest")
plt.tight_layout()
plt.show()

#2. XGBoost
xgb_importance = xgb.feature_importances_
xgb_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_importance
}).sort_values(by='Importance', ascending = False).head(15)
plt.figure(figsize=(8,6))
plt.barh(xgb_imp_df['Feature'], xgb_imp_df['Importance'], color = 'orange')
plt.gca().invert_yaxis()
plt.xlabel("Feature importance")
plt.title("Top 15 feature importance - XGBoost Classifier")
plt.tight_layout()
plt.show()

#Feature distribution - KDE
top_features = ['V14','V10','V4','V12','V11','V9']
plt.figure(figsize=(12,6))

for i, feature in enumerate(top_features,1):
    plt.subplot(2,3,i)
    sns.kdeplot(
        data=data,
        x=feature,
        hue="Class",
        fill=True,
        common_norm=False,
        palette={0:"green",1:'red'}
    )
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend(labels=['Normal','Fraud'])
plt.tight_layout()
plt.show()
