import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic data
dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
amounts = np.random.randint(100, 5000, size=100)
vendors = np.random.choice(['VendorA', 'VendorB', 'VendorC', 'VendorD'], 100)
statuses = np.random.choice(['Processed', 'Pending', 'Failed'], 100)
duplicates = np.random.choice([0, 1], 100, p=[0.9, 0.1])  # 10% chance of being a duplicate

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Amount': amounts,
    'Vendor': vendors,
    'Status': statuses,
    'Duplicate': duplicates
})

# Introduce anomalies: duplicate entries
for _ in range(10):
    idx = np.random.choice(df.index)
    df = df.append(df.loc[idx], ignore_index=True)

# Shuffle dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv('invoice_data.csv', index=False)
print("Data generated and saved to invoice_data.csv")
