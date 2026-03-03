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
import ROOT

def load_dataset():
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

    br_type = 'TISR3PI_SIG'
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
            X_train, y_train, X_val, y_val = load_dataset()
            print(f"X_train columns: {X_train.columns}")
            params = baye_opti(X_train, y_train) # Find best model parameters
            #params = set_model_params(X_train, y_train) # Initial model parameters
            params['early_stopping_rounds'] = 50 # Add early stop parameter (avoid early stop for model version saved in ROOT)
            #print("\nModel parameters:", params)

            # Create a model
            model = xgb.XGBClassifier(**params) 

            # Fit with the model
            model.fit(X_train, y_train,
                    eval_set = [(X_train, y_train), (X_val, y_val)],
                    verbose=False
            )

            # Save the model
            joblib.dump(model, f'{model_dir}/pi0_classifier_model_{br_type}.pkl')
            #print(f"Model is saved!")
            ROOT.TMVA.Experimental.SaveXGBoost(model, "BDT_pi0", rf"bdt_pi0_{br_type}.root", num_inputs=X_train.shape[1])
            print(f"Model saved successfully to bdt_pi0_{br_type}.root")