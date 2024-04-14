import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic data
dates = pd.date_range(start="2023-01-01", periods=200, freq='D')  # Increased the number of days
amounts = np.random.randint(100, 5000, size=200)  # More entries
vendors = np.random.choice(['VendorA', 'VendorB', 'VendorC', 'VendorD'], 200)
statuses = np.random.choice(['Processed', 'Pending', 'Failed'], 200)
duplicates = np.random.choice([0, 1], 200, p=[0.7, 0.3])  # Increased chance of duplicates

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Amount': amounts,
    'Vendor': vendors,
    'Status': statuses,
    'Duplicate': duplicates
})

# Introduce more anomalies: add duplicate entries with slight variations
for _ in range(30):  # More duplicates
    idx = np.random.choice(df.index[df['Duplicate'] == 1])  # Choose only from existing duplicates
    dup_entry = df.loc[idx].copy()
    dup_entry['Amount'] = dup_entry['Amount'] * np.random.normal(1.0, 0.05)  # Small variation in amount
    df = pd.concat([df, df.loc[[idx]]], ignore_index=True)

# Shuffle dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv('augmented_invoice_data.csv', index=False)
print("Augmented data generated and saved to augmented_invoice_data.csv")
