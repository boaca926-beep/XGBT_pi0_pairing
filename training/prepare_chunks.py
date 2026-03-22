# prepare_chunks.py - Run this ONCE before main_training.py
import numpy as np
import sys
import joblib
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import gc
from config import DATA_DIR, DATA_LARGE_DIR

def prepare_chunks(br_type, chunk_size=500000):
    """
    Split large dataset into chunks for external memory training.
    """
    input_data_dir = DATA_LARGE_DIR
    
    print(f"Preparing chunked data for {br_type}...")
    
    # Load original data
    print("Loading original data...")
    X_train = joblib.load(os.path.join(input_data_dir, f'X_train_{br_type}.pkl'))
    y_train = joblib.load(os.path.join(input_data_dir, f'y_train_{br_type}.pkl'))
    X_val = joblib.load(os.path.join(input_data_dir, f'X_val_{br_type}.pkl'))
    y_val = joblib.load(os.path.join(input_data_dir, f'y_val_{br_type}.pkl'))
    
    # Save feature names
    training_cols = [col for col in X_train.columns if col != 'is_pi0']
    joblib.dump(training_cols, os.path.join(input_data_dir, f'training_cols_{br_type}.pkl'))
    
    # Convert to numpy
    print("Converting to numpy...")
    X_train_np = X_train[training_cols].to_numpy()
    y_train_np = y_train.to_numpy().ravel()
    X_val_np = X_val[training_cols].to_numpy()
    y_val_np = y_val.to_numpy().ravel()
    
    # Free original data
    del X_train, y_train, X_val, y_val
    gc.collect()
    
    # Split training data
    n_samples = len(X_train_np)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    print(f"Splitting {n_samples} samples into {n_chunks} chunks...")
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_samples)
        
        np.save(os.path.join(input_data_dir, f'X_train_{br_type}_chunk_{i:04d}.npy'), 
                X_train_np[start:end])
        np.save(os.path.join(input_data_dir, f'y_train_{br_type}_chunk_{i:04d}.npy'), 
                y_train_np[start:end])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_chunks} chunks")
    
    # Split validation data
    n_val_samples = len(X_val_np)
    n_val_chunks = (n_val_samples + chunk_size - 1) // chunk_size
    
    print(f"Splitting {n_val_samples} validation samples into {n_val_chunks} chunks...")
    
    for i in range(n_val_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_val_samples)
        
        np.save(os.path.join(input_data_dir, f'X_val_{br_type}_chunk_{i:04d}.npy'), 
                X_val_np[start:end])
        np.save(os.path.join(input_data_dir, f'y_val_{br_type}_chunk_{i:04d}.npy'), 
                y_val_np[start:end])
    
    print(f"✓ Chunking complete!")
    print(f"  Training: {n_chunks} chunks")
    print(f"  Validation: {n_val_chunks} chunks")

if __name__ == '__main__':
    prepare_chunks('TCOMB', chunk_size=500000)