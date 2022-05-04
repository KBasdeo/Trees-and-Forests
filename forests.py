import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Using decision trees and random forests to classify whether a customer will pay back their loan in full

# Display settings for pandas dataframe output on pycharm
desired_width = 410
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)

# Storing loan data into a dataframe
loan_data = pd.read_csv('loan_data.csv')
print(loan_data.head())

# Count-plot displaying the relationship between the purpose of the loan and whether it was fully paid
sns.countplot(x=loan_data['purpose'], hue=loan_data['not.fully.paid'])
# Observing the relationship between a customer's fico score and their interest rate
sns.jointplot(x=loan_data['fico'], y=loan_data['int.rate'])
# plt.show()

# The purpose column in the dataset contains categorical data, so this should be converted to dummy data for model
cat_feats = ['purpose']
final_data = pd.get_dummies(loan_data, columns=cat_feats, drop_first=True)

# Setting up train/test split for ml model
X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Training and determining success of Decision Tree Model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Training and determining success of Random Forest Model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_predictions = rfc.predict(X_test)
print(confusion_matrix(y_test, rfc_predictions))
print(classification_report(y_test, rfc_predictions))
