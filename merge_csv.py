import os
import pandas as pd

dataset_folder = r"C:\Users\patel\OneDrive\Documents\Desktop\AI\CICIDS2017"
out_file = r"C:\Users\patel\OneDrive\Documents\Desktop\AI\CICIDS2017_merged.csv"

csv_files = sorted([os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".csv")])
print(f"Found {len(csv_files)} CSV files.")

first = True
for f in csv_files:
    print("Merging:", f)
    for chunk in pd.read_csv(f, chunksize=200000, low_memory=False):
        chunk.to_csv(out_file, mode="a", index=False, header=first)
        first = False

print("Merged dataset saved to:", out_file)
