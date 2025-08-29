import pandas as pd
import numpy as np

# Input merged CSV
IN = r"C:\Users\patel\OneDrive\Documents\Desktop\AI\CICIDS2017_merged.csv"
# Output cleaned CSV
OUT = r"C:\Users\patel\OneDrive\Documents\Desktop\AI\CICIDS2017_cleaned.csv"

# Columns to drop (identifiers, IPs, ports, timestamps)
drop_cols = ['Flow ID','Src IP','Dst IP','Source IP','Destination IP','Source Port','Destination Port','Timestamp']

first = True

# Process in chunks to save memory
for chunk in pd.read_csv(IN, chunksize=200000, low_memory=False):
    # Drop unwanted columns
    cols_to_drop = [c for c in drop_cols if c in chunk.columns]
    chunk = chunk.drop(columns=cols_to_drop)

    # Convert object columns (except Label) to numeric
    for c in chunk.select_dtypes(include=['object']).columns:
        if c.lower() != 'label':
            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')

    # Replace Inf and fill NaNs with median
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = chunk.select_dtypes(include=[np.number]).columns
    chunk[num_cols] = chunk[num_cols].fillna(chunk[num_cols].median())

    # Drop rows with missing Label
    if 'Label' in chunk.columns:
        chunk = chunk.dropna(subset=['Label'])

    # Save cleaned chunk
    chunk.to_csv(OUT, mode='a', index=False, header=first)
    first = False

print("Cleaned dataset saved as:", OUT)
