import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os

# Create dummy dataset
np.random.seed(42)
n_samples = 1000

# Features: Annual Income, Loan Amount, Credit Score, Employment Type, Years of Employment, Existing Credit Lines
annual_income = np.random.normal(60000, 20000, n_samples)
loan_amount = np.random.normal(15000, 8000, n_samples)
credit_score = np.random.normal(650, 80, n_samples)
employment_type = np.random.choice([0, 1, 2], n_samples) # 0: Salaried, 1: Self-Employed, 2: Freelancer
years_employment = np.random.normal(5, 3, n_samples).clip(0, 40)
existing_credit_lines = np.random.poisson(3, n_samples)

# Normalize/adjust arrays to realistic values
annual_income = np.clip(annual_income, 20000, 200000)
loan_amount = np.clip(loan_amount, 1000, 100000)
credit_score = np.clip(credit_score, 300, 850)

X = pd.DataFrame({
    'annual_income': annual_income,
    'loan_amount': loan_amount,
    'credit_score': credit_score,
    'employment_type': employment_type,
    'years_employment': years_employment,
    'existing_credit_lines': existing_credit_lines
})

# Generate target variable (Risk: 1 for default, 0 for paid)
# Lower income, higher loan, lower credit score -> higher risk
linear_comb = (
    -0.00005 * X['annual_income'] + 
    0.0001 * X['loan_amount'] - 
    0.01 * X['credit_score'] + 
    0.5 * X['employment_type'] - 
    0.1 * X['years_employment'] + 
    0.3 * X['existing_credit_lines']
)

probabilities = 1 / (1 + np.exp(-linear_comb))
# Shift probabilities to have a reasonable default rate
probabilities = probabilities - probabilities.mean() + 0.3
probabilities = np.clip(probabilities, 0.01, 0.99)

y = np.random.binomial(1, probabilities)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a pipeline with scaler and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate
train_acc = pipeline.score(X_train, y_train)
test_acc = pipeline.score(X_test, y_test)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")

# Save the model
model_path = 'model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)

print(f"Model saved to {model_path}")
