import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Show current directory to verify path
print("Current Working Directory:", os.getcwd())

# Step 1: Load Data
file_path = 'data/online_retail_ii.xlsx'
df = pd.read_excel(file_path)
print("Dataset Shape:", df.shape)
print("First few rows:")
print(df.head())

# Step 2: Data Cleaning
df = df[df['Country'] == 'United Kingdom']
df = df.dropna(subset=['Customer ID'])
df = df[df['Quantity'] > 0]
df['TotalPrice'] = df['Quantity'] * df['Price']

print("Cleaned Data Shape:", df.shape)
print("Cleaned Data Sample:")
print(df.head())

# Step 3: RFM Calculation
ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print("Reference Date (for Recency):", ref_date.date())

rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (ref_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print("\nRFM Table:")
print(rfm.head())

# Step 4: RFM Scoring
r_labels = range(5, 0, -1)
f_labels = m_labels = range(1, 6)

r_quartiles = pd.qcut(rfm['Recency'], q=5, labels=r_labels)
f_quartiles = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=f_labels)
m_quartiles = pd.qcut(rfm['Monetary'], q=5, labels=m_labels)

rfm['R'] = r_quartiles.astype(int)
rfm['F'] = f_quartiles.astype(int)
rfm['M'] = m_quartiles.astype(int)

rfm['RFM_Segment'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)

print("\nScored RFM Table:")
print(rfm.head())

# Step 5: Customer Segmentation
def segment_customer(row):
    if row['R'] >= 4 and row['F'] >= 4 and row['M'] >= 4:
        return 'Loyal Customers'
    elif row['R'] >= 4 and row['F'] >= 3:
        return 'Recent Engaged'
    elif row['R'] >= 3 and row['M'] >= 4:
        return 'Big Spenders'
    elif row['R'] >= 4:
        return 'Recent Customers'
    elif row['F'] >= 4:
        return 'Frequent Buyers'
    elif row['M'] >= 4:
        return 'High Value'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# Step 6: Visualize Segment Distribution
segment_counts = rfm['Segment'].value_counts()
print("\nCustomer Count per Segment:")
print(segment_counts)

plt.figure(figsize=(10, 6))
segment_counts.plot(kind='bar', color='skyblue')
plt.title('Customer Segments by Count')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


rfm.to_csv("outputs/rfm_scores.csv", index=False)


