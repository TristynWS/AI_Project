import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Define features and target
X = df.drop('Duplicate', axis=1)  # Make sure 'Duplicate' is the name of your target column
y = df['Duplicate']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForest model with a grid search over specified parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
clf = GridSearchCV(RandomForestClassifier(class_weight='balanced'), param_grid, cv=5)
clf.fit(X_train, y_train)

# Best model evaluation
print("Best model parameters:", clf.best_params_)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))
