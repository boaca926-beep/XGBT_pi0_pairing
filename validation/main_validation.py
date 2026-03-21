# Validation script
import os
import sys
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#import xgboost as xgb
from analysis.plots import plot_learning_curves, plot_roc
from metrics import eval_performance, event_performance

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have Qt installed
import matplotlib.pyplot as plt
plt.show(block=False)

from config import (
    DATA_DIR, DATA_LARGE_DIR, PLOT_DIR_VAL
)

def load_data():
    """
    Load validation data and models
    """

    # Load validation dataset
    X_val = joblib.load(os.path.join(input_data_dir, f'X_val_{br_nm}.pkl'))
    y_val = joblib.load(os.path.join(input_data_dir, f'y_val_{br_nm}.pkl'))
   
    # Load all_df
    all_df = joblib.load(os.path.join(input_data_dir, f'all_df_val_{br_nm}.pkl'))
    
    return X_val, y_val, all_df

if __name__ == '__main__':
    print(f"Validation ...")

    #input_data_dir = os.path.join(project_root, f'analysis/dataset')
    input_data_dir = DATA_LARGE_DIR
    phys_map = joblib.load(os.path.join(input_data_dir, f'phys_map.pkl'))
    
    print(phys_map)

    ## Load dataset
    phys_ch = ['TCOMB', 'combined']
    #phys_ch = ['TISR3PI_SIG', 'signal']
    #phys_ch = ['TETAGAM', 'signal']
    #data_type = 'TISR3PI_SIG' #'TETAGAM', 'TISR3PI_SIG', 'TKSL'd
    #input_data_dir = os.path.join(project_root, f'output_data_{input_str}')
    #input_data_dir = '../analysis/dataset'

    # Load phys_map
    #phys_map = joblib.load(os.path.join(input_data_dir, 'phys_map.pkl'))
    br_nm = phys_ch[0]
    info = phys_map.get(br_nm, "")
    print(info)
    br_title = info['br_title']

    X_val, y_val, all_df = load_data()
    model = joblib.load(os.path.join('../training/models', f'pi0_classifier_model_{br_nm}.pkl'))

    plot_dir = PLOT_DIR_VAL
    import shutil
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    features = X_val.columns
    #print(model.get_params())

    ## Evaluate validation set
    eval_performance(model, X_val, y_val)

    ## Feature importance -  check hat m_gg and opening_angle are top
    importance = model.feature_importances_
    for f, imp in zip(features, importance):
        print(f"    {f}: {imp:.03f}")

    ## Learning curves
    fig_learning = plot_learning_curves(model, rf'Learning Curve (validation, {br_title})')
    fig_learning.savefig(f'{plot_dir}/learning_curves_{br_nm}.png', dpi=300, bbox_inches='tight')

    ## Accuracy metrics, event basis
    score_list, var_list, var_str = event_performance(all_df, model)

    ## ROC plot
    fig_roc = plot_roc(score_list, rf'ROC Curve - $\pi^{0}$ Classifier (validation, {br_title})')
    fig_roc.savefig(f'{plot_dir}/roc_curv_{br_nm}.png', dpi=300, bbox_inches='tight')
    
    
    