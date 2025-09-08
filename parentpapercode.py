# Import packages
import pandas as pd
import sklearn as sk
import numpy as np

# Load data
GermanCredit = pd.read_csv(r"C:\Users\samar\Downloads\index.csv")
GermanCredit.head()
GermanCredit.columns

# Rename columns
GermanCredit.rename(columns = {'Duration of Credit (month)':'CreditDuration'
                              ,'Length of current employment':'CurrentEmploymentDuration'
                              ,'Duration in Current address':'CurrentAddressDuration'
                              ,'No of Credits at this Bank':'NoOfCreditsInBank'
                              ,'Type of apartment':'ApartmentType'
                              ,'Instalment per cent':'InstalmentPercentage'
                              ,'No of dependents': 'NoOfDependents'
                              ,'Most valuable available asset':'MostValuableAsset'
                              ,'Value Savings/Stocks': 'ValueSavingsOrStocks'
                              ,'Sex & Marital Status': 'SexAndMaritalStatus'}, inplace=True)

# Check for missing values
GermanCredit.isnull()
GermanCredit.isnull().any()
GermanCredit.shape

# Clean data
GC_with_Condition = GermanCredit.dropna(thresh=1)
GC_with_Condition.shape
GC_with_Condition.columns = GC_with_Condition.columns.str.replace(' ', '')  # Just renaming the column Names

# Explore data
GC_with_Condition['SexAndMaritalStatus'].value_counts()
GC_with_Condition['ApartmentType'].value_counts()
GC_with_Condition['ForeignWorker'].value_counts()
GC_with_Condition['MostValuableAsset'].value_counts()
GC_with_Condition['Occupation'].value_counts()
GC_with_Condition['ValueSavingsOrStocks'].value_counts()

# Split features and target
X = GC_with_Condition.iloc[:, 1:21]
Y = GC_with_Condition.iloc[:, 0]

# Split data
from sklearn.model_selection import train_test_split
GC_X_Train, GC_X_Test, GC_Y_Train, GC_Y_Test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
GC_X_Train = scaler.fit_transform(GC_X_Train)
GC_X_Test  = scaler.transform(GC_X_Test)

# Train model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)  # bumped from default to 1000
clf.fit(GC_X_Train, GC_Y_Train)

# Make predictions
predictions = clf.predict(GC_X_Test)

# Evaluate model
from sklearn.metrics import confusion_matrix
clf.score(GC_X_Test, GC_Y_Test)

confusion_matrix(GC_Y_Test, predictions)

y_score = clf.decision_function(GC_X_Test)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(GC_Y_Test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))