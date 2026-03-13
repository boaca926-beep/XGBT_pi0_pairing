# Training script
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from bayes_opti import baye_opti
import xgboost as xgb
import joblib
import random

def load_dataset(data_type):
    """
    Load train, val dataset
    """

    # Load X_train, y_train
    X_train = joblib.load(os.path.join(input_data_dir, f'X_train_{data_type}.pkl'))
    y_train = joblib.load(os.path.join(input_data_dir, f'y_train_{data_type}.pkl'))

    # Load X_val, y_val
    X_val = joblib.load(os.path.join(input_data_dir, f'X_val_{data_type}.pkl'))
    y_val = joblib.load(os.path.join(input_data_dir, f'y_val_{data_type}.pkl'))
 
    return X_train, y_train, X_val, y_val

if __name__ == '__main__':

    print(f"Train combined dataset ...")

    input_data_dir = os.path.join(project_root, f'analysis/dataset')
    os.makedirs('models', exist_ok=True)  

    ## Bayesian optimization (look for best model parameters)
    data_type = 'TCOMB'
    X_train, y_train, X_val, y_val = load_dataset(data_type)
    
    params = baye_opti(X_train, y_train) # Find best model parameters
    #params = set_model_params(X_train, y_train) # Initial model parameters
    params['early_stopping_rounds'] = 50 # Add early stop parameter
    #print("\nModel parameters:", params)

    # Create a model
    model = xgb.XGBClassifier(**params) 

    # Fit with the model
    model.fit(X_train, y_train,
              eval_set = [(X_train, y_train), (X_val, y_val)],
              verbose=False
              )
    
    # Save the model
    joblib.dump(model, f'models/pi0_classifier_model_{data_type}.pkl')
    #print(f"Model is saved!")