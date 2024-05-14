# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Load Data

# Data Paths
RAW_DATA_PATH = './data-for-project-1/raw_data.csv'
VALIDATION_DATA_PATH = './data-for-project-1/validation.csv'

# Loading data
raw = pd.read_csv(RAW_DATA_PATH)
validation = pd.read_csv(VALIDATION_DATA_PATH)

# Creating copies for manipulation
train_original = raw.copy()
test_original = validation.copy()

# Preprocess Data
# Fill missing values
def fill_missing_values(data):
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)

fill_missing_values(raw)
fill_missing_values(validation)

# Feature engineering
raw['Total_Income'] = raw['ApplicantIncome'] + raw['CoapplicantIncome']
validation['Total_Income'] = validation['ApplicantIncome'] + validation['CoapplicantIncome']
raw['EMI'] = raw['LoanAmount'] / raw['Loan_Amount_Term']
validation['EMI'] = validation['LoanAmount'] / validation['Loan_Amount_Term']
raw['Balance_Income'] = raw['Total_Income'] - (raw['EMI'] * 1000)
validation['Balance_Income'] = validation['Total_Income'] - (validation['EMI'] * 1000)

# Drop unneeded columns
columns_to_drop = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
raw.drop(columns=columns_to_drop, axis=1, inplace=True)
validation.drop(columns=columns_to_drop, axis=1, inplace=True)

# Model Building and Prediction
# Data preparation for model
X = pd.get_dummies(raw.drop('Loan_Status', axis=1))
y = raw['Loan_Status']
validation_processed = pd.get_dummies(validation)
validation_processed = validation_processed.reindex(columns=X.columns, fill_value=0)

# Model training and evaluation
def train_and_evaluate(X, y, validation):
    x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=1)
    model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=41)
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_cv, model.predict(x_cv))
    print(f"Validation Accuracy: {accuracy}")
    
    # Predict on validation data
    pred_validation = model.predict(validation)
    return pred_validation

predictions = train_and_evaluate(X, y, validation_processed)

# Output Predictions
# Save the predictions to a CSV file
output = pd.DataFrame({'Loan_ID': test_original['Loan_ID'], 'Loan_Status': predictions})
output.to_csv('predictions.csv', index=False)
print("Predictions are saved to predictions.csv")
