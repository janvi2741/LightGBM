import pandas as pd
import numpy as np
import os
import glob

def check_original_files():
    """Check the original CSV files for class distribution"""
    print("=== DIAGNOSING ORIGINAL FILES ===")
    dataset_folder = "CICIDS2017"
    csv_files = glob.glob(os.path.join(dataset_folder, "*.csv"))
    
    total_benign = 0
    total_attacks = 0
    
    for file in csv_files:
        print(f"\nChecking {os.path.basename(file)}...")
        try:
            # Read first few rows to understand structure
            df_sample = pd.read_csv(file, nrows=1000, low_memory=False)
            df_sample.columns = df_sample.columns.str.strip()
            
            # Find label column - check multiple possible names
            label_col = None
            possible_labels = ['Label', 'label', ' Label', 'Label ', ' Label ']
            for possible in possible_labels:
                if possible in df_sample.columns:
                    label_col = possible
                    break
            
            # If not found, print all columns for debugging
            if not label_col:
                print(f"  Available columns: {list(df_sample.columns)}")
                # Try to find any column with 'label' in the name (case insensitive)
                for col in df_sample.columns:
                    if 'label' in str(col).lower():
                        label_col = col
                        break
            
            if label_col:
                print(f"  Label column: {label_col}")
                # Check full file for class distribution
                label_counts = {}
                chunk_size = 50000
                
                for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
                    chunk.columns = chunk.columns.str.strip()
                    if label_col in chunk.columns:
                        chunk_counts = chunk[label_col].value_counts()
                        for label, count in chunk_counts.items():
                            label_counts[label] = label_counts.get(label, 0) + count
                
                print(f"  Class distribution: {label_counts}")
                
                # Count benign vs attacks
                for label, count in label_counts.items():
                    if str(label).upper() == 'BENIGN':
                        total_benign += count
                    else:
                        total_attacks += count
            else:
                print(f"  No label column found in {file}")
                
        except Exception as e:
            print(f"  Error reading {file}: {e}")
    
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total BENIGN samples: {total_benign}")
    print(f"Total ATTACK samples: {total_attacks}")
    print(f"Total samples: {total_benign + total_attacks}")
    
    return total_benign, total_attacks

def create_balanced_dataset():
    """Create a properly balanced dataset from original files"""
    print("\n=== CREATING BALANCED DATASET ===")
    
    dataset_folder = "CICIDS2017"
    csv_files = glob.glob(os.path.join(dataset_folder, "*.csv"))
    
    benign_samples = []
    attack_samples = []
    
    # Collect samples from each file
    for file in csv_files:
        print(f"Processing {os.path.basename(file)}...")
        
        try:
            chunk_size = 50000
            for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
                chunk.columns = chunk.columns.str.strip()
                
                # Find label column
                label_col = None
                possible_labels = ['Label', 'label', ' Label', 'Label ', ' Label ']
                for possible in possible_labels:
                    if possible in chunk.columns:
                        label_col = possible
                        break
                
                if not label_col:
                    for col in chunk.columns:
                        if 'label' in str(col).lower():
                            label_col = col
                            break
                
                if label_col and label_col in chunk.columns:
                    # Separate benign and attack samples
                    benign_chunk = chunk[chunk[label_col].str.upper() == 'BENIGN']
                    attack_chunk = chunk[chunk[label_col].str.upper() != 'BENIGN']
                    
                    if len(benign_chunk) > 0:
                        # Sample up to 10000 benign per file
                        sample_size = min(len(benign_chunk), 10000)
                        benign_samples.append(benign_chunk.sample(n=sample_size, random_state=42))
                    
                    if len(attack_chunk) > 0:
                        # Take all attack samples (they're usually fewer)
                        attack_samples.append(attack_chunk)
                        
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Combine all samples
    print("Combining samples...")
    all_benign = pd.concat(benign_samples, ignore_index=True) if benign_samples else pd.DataFrame()
    all_attacks = pd.concat(attack_samples, ignore_index=True) if attack_samples else pd.DataFrame()
    
    print(f"Collected {len(all_benign)} benign samples")
    print(f"Collected {len(all_attacks)} attack samples")
    
    if len(all_attacks) == 0:
        print("ERROR: No attack samples found! Check your data files.")
        return False
    
    # Balance the dataset
    min_samples = min(len(all_benign), len(all_attacks))
    balanced_samples = min(min_samples, 50000)  # Limit to 50k per class
    
    print(f"Creating balanced dataset with {balanced_samples} samples per class...")
    
    final_benign = all_benign.sample(n=balanced_samples, random_state=42)
    final_attacks = all_attacks.sample(n=balanced_samples, random_state=42)
    
    # Combine and shuffle
    balanced_df = pd.concat([final_benign, final_attacks], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Clean the dataset
    print("Cleaning dataset...")
    
    # Drop unwanted columns
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Source IP', 'Destination IP', 
                 'Source Port', 'Destination Port', 'Timestamp']
    cols_to_drop = [c for c in drop_cols if c in balanced_df.columns]
    balanced_df = balanced_df.drop(columns=cols_to_drop)
    
    # Convert numeric columns
    label_col = None
    for col in balanced_df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    for col in balanced_df.columns:
        if col != label_col and balanced_df[col].dtype == 'object':
            balanced_df[col] = pd.to_numeric(balanced_df[col], errors='coerce')
    
    # Handle infinite values and NaNs
    balanced_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = balanced_df.select_dtypes(include=[np.number]).columns
    balanced_df[numeric_cols] = balanced_df[numeric_cols].fillna(balanced_df[numeric_cols].median())
    
    # Save the balanced dataset
    output_file = "CICIDS2017_cleaned.csv"
    balanced_df.to_csv(output_file, index=False)
    
    print(f"Balanced dataset saved to: {output_file}")
    print(f"Final dataset shape: {balanced_df.shape}")
    
    # Verify class distribution
    if label_col:
        final_counts = balanced_df[label_col].value_counts()
        print(f"Final class distribution: {final_counts.to_dict()}")
    
    return True

if __name__ == "__main__":
    # Step 1: Diagnose original files
    total_benign, total_attacks = check_original_files()
    
    # Step 2: Create balanced dataset if attacks exist
    if total_attacks > 0:
        success = create_balanced_dataset()
        if success:
            print("\n✅ Dataset successfully created with balanced classes!")
        else:
            print("\n❌ Failed to create balanced dataset")
    else:
        print("\n❌ No attack samples found in original files!")
        print("Please check if your CICIDS2017 files contain attack data.")
