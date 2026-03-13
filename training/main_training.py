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

# Define the patch function
def patched_get_basescore(model):
    config_str = model.get_booster().save_config()
    config = json.loads(config_str)
    base_score_str = config["learner"]["learner_model_param"]["base_score"]
    # Remove brackets if present
    base_score_str = base_score_str.strip('[]')
    return float(base_score_str)

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
    os.makedirs(model_dir, exist_ok=True)

    input_data_dir = os.path.join(project_root, f'analysis/dataset')
    phys_map = joblib.load(os.path.join(input_data_dir, f'phys_map.pkl'))
    print(phys_map)

    br_type = 'TCOMB' #'TETAGAM', TISR3PI_SIG', 'TCOMB'
    info = phys_map[br_type]
    info_title = info['br_title']
    info_category = info['category']
    print(f"Inspecting dataset of data_type: {br_type}; title: {info_title}; category: {info_category}")  
    #print(info)
    
    for data_type, info in phys_map.items():
        br_title = info['br_title']
        category = info['category']
        #print(f"Inspecting dataset {data_type}; {br_nm}; {br_title}; {category}")  
        print(info)

        if (data_type == br_type):
            ## Bayesian optimization (look for best model parameters)
            X_train, y_train, X_val, y_val = load_dataset(br_type)
            print(f"X_train columns: {X_train.columns}")
            print(f"Number of features in the training: {len(X_train.columns.tolist())}")

            params = baye_opti(X_train, y_train) # Find best model parameters
            #params = set_model_params(X_train, y_train) # Initial model parameters
            params['early_stopping_rounds'] = 50 # Add early stop parameter (avoid early stop for model version saved in ROOT)
            #print("\nModel parameters:", params)

            # CONVERT TO NUMPY ARRAYS (THIS IS THE KEY FIX!)
            # Define training columns (exclude target)
            # To avoid problems in saving models with .root file style
            training_cols = [col for col in X_train.columns if col != 'is_pi0']

            X_train_np = X_train[training_cols].to_numpy()
            X_val_np = X_val[training_cols].to_numpy()
            y_train_np = y_train.to_numpy().ravel()  # Ensure 1D
            y_val_np = y_val.to_numpy().ravel()

            # Create a model
            model = xgb.XGBClassifier(**params) 

            # Fit with the model
            model.fit(X_train_np, y_train_np,
                    eval_set = [(X_train_np, y_train_np), (X_val_np, y_val_np)],
                    #verbose=False
                    verbose=50     
            )

            # Save the model
            joblib.dump(model, f'{model_dir}/pi0_classifier_model_{br_type}.pkl')
            #print(f"Model is saved!")

            # Get the booster
            booster = model.get_booster()

            # Import the module and patch it
            import ROOT._pythonization._tmva._tree_inference as tree_inference
            tree_inference.get_basescore = patched_get_basescore

            # Try saving with booster instead of model
            ROOT.TMVA.Experimental.SaveXGBoost(
                model,  # ← This is XGBClassifier, which has .objective attribute
                "BDT_pi0", 
                f"{model_dir}/bdt_pi0_{br_type}.root", 
                num_inputs=X_train_np.shape[1]
            )

            # Check model root files in terminal:
            # > ls -la bdt_pi0_*.root
            # import ROOT
            # f = ROOT.TFile.Open(f"bdt_pi0_{br_type}.root")
            # f.ls()

            print(f"✓ Model saved to bdt_pi0_{br_type}.root")