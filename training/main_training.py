# Training script
import uproot
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from bayes_opti import baye_opti
import joblib
import xgboost as xgb # xgboost must be imported before ROOT to prevent a crash caused by conflicting C++ std::regex symbols
import ROOT # This order is important!
import json

import multiprocessing
import psutil
import time
import gc

from config import (
    DATA_DIR,
    patched_get_basescore
)

'''
# Check saved files
ls -la models/

# View metrics
cat models/metrics_TCOMB.json | jq '.auc'

# Test ROOT file
root -l models/bdt_pi0_TCOMB.root -e ".ls"
'''

'''
In-memory approach is much faster, but barely works with large dataset 
'''

def load_dataset(br_type):
    """
    Load train, val dataset
    """

    # Load X_train, y_train
    X_train = joblib.load(os.path.join(input_data_dir, f'X_train_{br_type}.pkl'))
    y_train = joblib.load(os.path.join(input_data_dir, f'y_train_{br_type}.pkl'))

    # Load X_val, y_val
    X_val = joblib.load(os.path.join(input_data_dir, f'X_val_{br_type}.pkl'))
    y_val = joblib.load(os.path.join(input_data_dir, f'y_val_{br_type}.pkl'))
 
    return X_train, y_train, X_val, y_val

if __name__ == '__main__':

    print(f"Train individual signal physical channels ...")
    # Create ooutput directory
    model_dir = "./models" 

    # With this (always fresh):
    import shutil
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    os.makedirs(model_dir, exist_ok=True)

    input_data_dir = DATA_DIR
    #input_data_dir = os.path.join(project_root, f'{analysis/dataset}')
    phys_map = joblib.load(os.path.join(input_data_dir, f'phys_map.pkl'))
    print(phys_map)

    br_type = 'TCOMB' #'TETAGAM', TISR3PI_SIG', 'TCOMB'
    info = phys_map[br_type]
    info_title = info['br_title']
    info_category = info['category']
    print(f"Inspecting dataset of data_type: {br_type}; title: {info_title}; category: {info_category}")  
    #print(info)
    
    print("\n" + "="*60)
    print("STEP 1: BAYESIAN OPTIMIZATION")
    print("="*60)

    ## Bayesian optimization (look for best model parameters)
    X_train, y_train, X_val, y_val = load_dataset(br_type)
    print(f"X_train columns: {X_train.columns}")
    print(f"Number of features in the training: {len(X_train.columns.tolist())}")

    optimized_params = baye_opti(X_train, y_train) # Find best model parameters
    #params['nthread'] = -1 # Use all avaliable threads for faster training
    #params = set_model_params(X_train, y_train) # Initial model parameters
    #params['early_stopping_rounds'] = 50 # Add early stop parameter (avoid early stop for model version saved in ROOT)
    #print("\nModel parameters:", params)    # ✅ BEST: Simple but informative check

    #Set fixed parameters that shouldn't be optimized or need specific values
    params = {
         'nthread': -1,                      # Use all available threads
         'tree_method': 'hist',              # Histogram-based algorithm
         'early_stopping_rounds': 50,        # Early stopping
         'eval_metric': ['auc', 'error'],               # Evaluation metric
         'verbosity': 1,                     # Show progress
         # Keep the optimized parameters from Bayesian optimization
         'max_depth': optimized_params.get('max_depth', 10),
         'learning_rate': optimized_params.get('learning_rate', 0.1),
         'subsample': optimized_params.get('subsample', 0.8),
         'colsample_bytree': optimized_params.get('colsample_bytree', 0.8),
         'min_child_weight': optimized_params.get('min_child_weight', 1),
         'gamma': optimized_params.get('gamma', 0),
         'reg_alpha': optimized_params.get('reg_alpha', 0),
         'reg_lambda': optimized_params.get('reg_lambda', 1)
         }
    
    print("\n" + "="*50)
    print("THREAD CONFIGURATION CHECK")
    print("="*50)
    print(f"CPU cores available: {multiprocessing.cpu_count()}")
    print(f"XGBoost nthread setting: {params.get('nthread')}")
    print(f"XGBoost version: {xgb.__version__}")

    print(f"\n{'='*60}")
    print(f"STEP 2. TRAIN BEST MODEL")
    print(f"{'='*60}")    

    # CONVERT TO NUMPY ARRAYS (THIS IS THE KEY FIX!)
    # Define training columns (exclude target)
    # To avoid problems in saving models with .root file style
    
    training_cols = [col for col in X_train.columns if col != 'is_pi0']

    X_train_np = X_train[training_cols].to_numpy()
    X_val_np = X_val[training_cols].to_numpy()
    y_train_np = np.asarray(y_train).ravel()  # FIX: Use np.asarray instead of to_numpy().ravel()
    y_val_np = np.asarray(y_val).ravel()      # FIX: Use np.asarray instead of to_numpy().ravel()
    #y_train_np = y_train.to_numpy().ravel()  # Ensure 1D
    #y_val_np = y_val.to_numpy().ravel()

    # Create a model
    model = xgb.XGBClassifier(**params) 
    print(f"Model will use: {model.get_params().get('nthread', 'default')} threads")
    
    # Quick CPU monitor during training (non-intrusive)
    print("\nTraining started - monitoring CPU for 5 seconds...")
    cpu_percents = []
    start_time = time.time()

    # Fit with the model
    model.fit(X_train_np, y_train_np,
              eval_set = [(X_train_np, y_train_np), (X_val_np, y_val_np)],
              #eval_metric='auc', # Required for proper early stopping
              #verbose=False
              verbose=50     
              )

    training_time = time.time() - start_time
    
    # Add this after training to see CPU usage:
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    print(f"\nCPU usage per core during training: {cpu_percent}")
    print(f"Average CPU usage: {sum(cpu_percent)/len(cpu_percent):.1f}%")
    print(f"Peak threads used: {psutil.Process().num_threads()}")
    print(f"Training time: {training_time/60:.1f} minutes")

    print("="*50)

    # Save the model
    joblib.dump(model, f'{model_dir}/pi0_classifier_model_{br_type}.pkl')
    print(f"Model saved to {model_dir}/pi0_classifier_model_{br_type}.pkl")
    #print(f"Model is saved!")

    print(f"\n{'='*60}")
    print(f"STEP 3: SAVE METRICS")
    print(f"{'='*60}") 
    # Save metrics
    from sklearn.metrics import roc_auc_score
    y_pred = model.predict_proba(X_val_np)[:, 1]
    auc_score = roc_auc_score(y_val_np, y_pred)
            
    metrics = {
         'auc': auc_score,
         'best_iteration': model.best_iteration,
         'best_score': model.best_score,
         'params': params,
         'n_features': len(training_cols)
         }
            
    with open(f'{model_dir}/metrics_{br_type}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {model_dir}/metrics_{br_type}.json")

    # Get the booster
    booster = model.get_booster()

    # Import the module and patch it
    import ROOT._pythonization._tmva._tree_inference as tree_inference
    tree_inference.get_basescore = patched_get_basescore

    # Try saving with booster instead of model
    try:
        ROOT.TMVA.Experimental.SaveXGBoost(
            model,  # ← This is XGBClassifier, which has .objective attribute
            "BDT_pi0", 
            f"{model_dir}/bdt_pi0_{br_type}.root", 
            num_inputs=X_train_np.shape[1]
        )
        print(f"✓ Model saved to {model_dir}/bdt_pi0_{br_type}.root")
    except Exception as e:
        print(f"✗ Failed to save ROOT model: {e}")

    # Memory cleanup
    del X_train_np, X_val_np, y_train_np, y_val_np
    gc.collect()
            
    print(f"\n✓ Training completed for {br_type}")

    # Check model root files in terminal:
    # > ls -la bdt_pi0_*.root
    # import ROOT
    # f = ROOT.TFile.Open(f"bdt_pi0_{br_type}.root")
    # f.ls()