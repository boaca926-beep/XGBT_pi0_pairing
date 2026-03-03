# Validation script
import os
import sys
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#import xgboost as xgb
from analysis.plots import plot_learning_curves, plot_roc
from metrics import eval_performance, event_performance

r'''
def load_data_model():
    """
    Load validation data from folder: validation_data
    Load training data from folder: training_data
    Load model from folder: models
    """

    # Data and model folders
    val_dir = os.path.join(project_root, 'validation_data')
    train_dir = os.path.join(project_root, 'training_data')
    model_dir = os.path.join(project_root, 'models')

    # Get validation file
    X_val = joblib.load(os.path.join(val_dir, 'X_val.pkl'))
    y_val = joblib.load(os.path.join(val_dir, 'y_val.pkl'))

    # Get training file
    X_train = joblib.load(os.path.join(train_dir, 'X_train.pkl'))
    y_train = joblib.load(os.path.join(train_dir, 'y_train.pkl'))

    # Load feature names
    with open(os.path.join(val_dir, 'feature_name.txt'), 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
        print(feature_names)
    features = X_val.columns
    
    # Load model
    model = joblib.load(os.path.join(model_dir, 'pi0_classifier_model.pkl'))

    return X_val, y_val, X_train, y_train, features, model
'''

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

    ## Load dataset
    data_type = 'ksl' #'TETAGAM', 'TISR3PI_SIG', 'TKSL'd
    #input_data_dir = os.path.join(project_root, f'output_data_{input_str}')
    input_data_dir = '../analysis/dataset'

    # Load phys_map
    phys_map = joblib.load(os.path.join(input_data_dir, 'phys_map_indiv.pkl'))
    phys_ch = phys_map.get(data_type, "")
    print(phys_ch)
    br_nm = phys_ch['br_nm']
    br_title = phys_ch['br_title']

    X_val, y_val, all_df = load_data()

    plot_dir = f'plots_val_{br_nm}'
    os.makedirs(plot_dir, exist_ok=True)

    features = X_val.columns
    #print(model.get_params())

    ## Load model
    model = joblib.load(os.path.join('../training/models', f'pi0_classifier_model_{br_nm}.pkl'))
    
    ## Evaluate validation set
    eval_performance(model, X_val, y_val)

    ## Feature importance -  check hat m_gg and opening_angle are top
    importance = model.feature_importances_
    for f, imp in zip(features, importance):
        print(f"    {f}: {imp:.03f}")

    ## Learning curves
    fig_learning = plot_learning_curves(model, rf'Model Validation (validation, {br_title})')
    fig_learning.savefig(f'./{plot_dir}/learning_curves_{br_nm}.png', dpi=300, bbox_inches='tight')

    ## Accuracy metrics, event basis
    score_list, var_list, var_str = event_performance(all_df, model)

    ## ROC plot
    fig_roc = plot_roc(score_list, rf'ROC Curve - $\pi^{0}$ Classifier (validation, {br_title})')
    fig_roc.savefig(f'./{plot_dir}/roc_curv_{br_nm}.png', dpi=300, bbox_inches='tight')

    
    