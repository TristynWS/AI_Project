import pandas as pd

# Load the dataset
df = pd.read_csv('augmented_invoice_data.csv')

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Assuming 'Status' is the problematic feature
df = pd.get_dummies(df, columns=['Status'])

# Extract day of the week and month from the 'Date' column
df['Day_of_Week'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
df['Month'] = df['Date'].dt.month  # January=1, December=12

# Check the first few rows to confirm the new columns are correct
print("Data with new features:")
print(df.head())

# Continue with any other preprocessing steps you need
# Example: Normalize the 'Amount' column
df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

# Save the preprocessed data back to a CSV file
df.to_csv('preprocessed_data.csv', index=False)

print("Preprocessing done. Data saved to preprocessed_data.csv")
