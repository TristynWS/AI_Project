import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('preprocessed_data.csv')

# Normalize features
scaler = StandardScaler()
numeric_features = ['Amount', 'Month', 'Day_of_Week']  # Update based on your actual numeric features
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Generate polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(df[numeric_features])

# Define features and target
X = np.hstack((X_poly, pd.get_dummies(df.drop(numeric_features + ['Duplicate'], axis=1)).values))  # Combining polynomial and categorical
y = df['Duplicate']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with SMOTE and XGBClassifier
pipeline = IMBPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Define the parameter grid
param_grid = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__gamma': [0.0, 0.1, 0.2],
    'classifier__subsample': [0.7, 0.8, 0.9],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1]
}

# Randomized search for the best parameters
random_search = RandomizedSearchCV(pipeline, param_grid, n_iter=50, scoring='roc_auc', cv=5, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# Output the best parameters and their corresponding score
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)

# Using the best estimator from the random search
best_model = random_search.best_estimator_

# Predictions and evaluation
predictions = best_model.predict(X_test)
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("ROC AUC Score:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve (AUC = %.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
plt.show()
