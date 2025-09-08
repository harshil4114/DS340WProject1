# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score

# Load data
GermanCredit = pd.read_csv(r"C:\Users\samar\Downloads\index.csv")

# Rename columns
GermanCredit.rename(columns = {'Duration of Credit (month)':'CreditDuration',
                              'Length of current employment':'CurrentEmploymentDuration',
                              'Duration in Current address':'CurrentAddressDuration',
                              'No of Credits at this Bank':'NoOfCreditsInBank',
                              'Type of apartment':'ApartmentType',
                              'Instalment per cent':'InstalmentPercentage',
                              'No of dependents': 'NoOfDependents',
                              'Most valuable available asset':'MostValuableAsset',
                              'Value Savings/Stocks': 'ValueSavingsOrStocks',
                              'Sex & Marital Status': 'SexAndMaritalStatus'}, inplace=True)

# Clean data
GC_with_Condition = GermanCredit.dropna(thresh=1)
GC_with_Condition.columns = GC_with_Condition.columns.str.replace(' ', '')

# Split features and target
X = GC_with_Condition.iloc[:, 1:21]  # features
Y = GC_with_Condition.iloc[:, 0]     # target

# Three-way split: Train (70%), Test (20%), Validation (10%)
RANDOM_STATE = 42

# First split: 80% temp + 20% test
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=RANDOM_STATE, stratify=Y
)

# Second split: 70% train + 10% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.125, random_state=RANDOM_STATE, stratify=y_train_temp
)

print("Dataset splits:")
print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")
print(f"Val:   {X_val.shape}")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Train model
clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
clf.fit(X_train_sc, y_train)

# Evaluate on test set
test_acc = clf.score(X_test_sc, y_test)
y_test_pred = clf.predict(X_test_sc)
y_test_score = clf.decision_function(X_test_sc)

test_cm = confusion_matrix(y_test, y_test_pred)
test_ap = average_precision_score(y_test, y_test_score)

print("\n=== TEST RESULTS ===")
print(f"Accuracy: {test_acc:.4f}")
print(f"PR-AUC: {test_ap:.4f}")
print(f"Confusion Matrix:\n{test_cm}")
print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred)}")