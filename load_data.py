import pandas as pd
import os

# Dataset folder path
dataset_path = r"C:\Users\patel\OneDrive\Documents\Desktop\AI\CICIDS2017"

# List all CSV files in dataset folder
csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

print("Available CSV files:")
for f in csv_files:
    print(f)

# Example: Load one dataset (change filename if needed)
file_to_load = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
data = pd.read_csv(os.path.join(dataset_path, file_to_load))

print("\nLoaded dataset:", file_to_load)
print("Shape:", data.shape)
print(data.head())
