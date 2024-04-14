import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Update the feature set to include the new one-hot encoded columns
feature_columns = [col for col in df.columns if col.startswith('Amount') or col.startswith('Status_')]  # Adjust as necessary
X = df[feature_columns]
y = df['Duplicate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, predictions))
