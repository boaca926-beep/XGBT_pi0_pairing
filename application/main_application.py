# Application

import pandas as pd
import numpy as np
import os
import sys
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from analysis.plots import plot_var_score, plot_roc, plot_nm
from validation.metrics import event_performance



def load_data_model():
    """
    Load test data from folder: test_data
    Load model from folder: models
    """

    # Load all_df, X_test, y_test
    all_df = joblib.load(os.path.join(input_data_dir, f'all_df_{data_type}.pkl'))
    X_test = joblib.load(os.path.join(input_data_dir, f'X_test_{data_type}.pkl'))
    y_test = joblib.load(os.path.join(input_data_dir, f'y_test_{data_type}.pkl'))

    # Load model
    model = joblib.load(os.path.join(input_model_dir, f'pi0_classifier_model_{data_type}.pkl'))
    
    return all_df, model, X_test, y_test
    
if __name__ == '__main__':

    print(f"Application on test dataset...")
    
    input_data_dir = os.path.join(project_root, f'analysis/dataset')
    input_model_dir = os.path.join(project_root, f'training/models')

    #category_type = ['indiv', 'signal']
    category_type = ['combined', 'combined']

    phys_map = joblib.load(os.path.join(input_data_dir, f'phys_map_{category_type[0]}.pkl'))
    print(phys_map)
    

    # Create output folder
    plot_dir = rf'./plots'
    os.makedirs(plot_dir , exist_ok=True)
    
    for data_type, info in phys_map.items():
        br_nm = info['br_nm']
        br_title = info['br_title']
        category = info['category']
        #print(f"Inspecting dataset {data_type}; {br_nm}; {br_title}; {category}")  
        print(info)

        if (category == category_type[1]):

            # Load test dataset and all_df
            all_df, model, X_test, y_test = load_data_model()

            # Selection cut (chi2, E_dela, opening angle, beta)

            # Check kine

            ## Plot confusion matrix
            fig_cm = plot_nm(X_test, y_test, model, br_title)
            fig_cm.savefig(f'./{plot_dir}/cm_{data_type}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_cm)

            ## Accuracy metrics, event basis
            score_list, var_list, var_str = event_performance(all_df, model)

            fig_var_score = plot_var_score(var_list, score_list, var_str, f"Mass and Score (test, {br_title})")
            fig_var_score.savefig(f'./{plot_dir}/pi0_mass_score_{data_type}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_var_score)

            ## ROC plot
            fig_roc = plot_roc(score_list, rf'ROC Curve - $\pi^{0}$ Classifier (test, {br_title})')
            fig_roc.savefig(f'./{plot_dir}/roc_curv_{data_type}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_roc)

            ## Plot kine. var after the pi0 identification

        else:
            print("No true labels")

   