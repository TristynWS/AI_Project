import pandas as pd

# Load the dataset
df = pd.read_csv('augmented_invoice_data.csv')

# Check the first few rows of the dataframe
print("Initial Data:")
print(df.head())

# Handle duplicates: mark them if not already marked
if 'Duplicate' not in df.columns:
    df['Duplicate'] = df.duplicated(subset=['Date', 'Amount', 'Vendor'], keep=False).astype(int)

# Replace status strings with numerical codes
status_mapping = {'Processed': 1, 'Pending': 0, 'Failed': -1}
df['Status'] = df['Status'].map(status_mapping)

# Normalize the 'Amount' column
df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

# Save the preprocessed data
df.to_csv('preprocessed_augmented_invoice_data.csv', index=False)

print("Preprocessing done. Data saved to preprocessed_data.csv")
