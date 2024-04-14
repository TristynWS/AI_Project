import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('invoice_data.csv')

# Ensure the 'Date' column is in datetime format and extract features
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day_of_Week'] = df['Date'].dt.dayofweek

# Drop the original 'Date' column if it's no longer needed
df.drop(['Date'], axis=1, inplace=True)

# Encode categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=['Status', 'Vendor'], drop_first=True)

# Scale numeric features
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Save the preprocessed data
df.to_csv('preprocessed_data.csv', index=False)
