import zipfile
import os

# Path to your zip file
zip_path = "archive.zip"
extract_path = "CICIDS2017"

# Extract zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"✅ Dataset extracted to: {os.path.abspath(extract_path)}")
