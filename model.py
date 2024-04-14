import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('invoice_data.csv')

# Preprocess the data
# Convert 'Date' to datetime if not already and extract useful features
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day_of_Week'] = df['Date'].dt.dayofweek
df.drop(['Date'], axis=1, inplace=True)  # Drop the Date column after extracting features

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['Status', 'Vendor'], drop_first=True)

# Normalize numeric features
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Define features and target
X = df.drop('Duplicate', axis=1)  # Assuming 'Duplicate' is your target variable
y = df['Duplicate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and configure the XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(len(y[y==0])/len(y[y==1])))

# Set up hyperparameter grid for tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0.0, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

# Grid search to find the best parameters
grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
grid.fit(X_train, y_train)

# Output the best parameters and score
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

# Using the best estimator found
model = grid.best_estimator_

# Predictions and evaluations
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
pr_auc = auc(recall, precision)
plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
