import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the preprocessed data
df = pd.read_csv('preprocessed_augmented_invoice_data.csv')

# Prepare the data for modeling
X = df[['Amount', 'Status']]  # Features
y = df['Duplicate']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, predictions))
