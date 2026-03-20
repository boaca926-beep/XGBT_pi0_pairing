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


def load_data_model(data_type):
    """
    Load test data from folder: test_data
    Load model from folder: models
    """

    # Load all_df, X_test, y_test
    all_df = joblib.load(os.path.join(input_data_dir, f'all_df_{data_type}.pkl'))
    all_df_test = joblib.load(os.path.join(input_data_dir, f'all_df_test_{data_type}.pkl'))

    X_test = joblib.load(os.path.join(input_data_dir, f'X_test_{data_type}.pkl'))
    y_test = joblib.load(os.path.join(input_data_dir, f'y_test_{data_type}.pkl'))

    # Load model
    model = joblib.load(os.path.join(input_model_dir, f'pi0_classifier_model_{data_type}.pkl'))
    
    return all_df, all_df_test, model, X_test, y_test

r'''
def event_wise_prediction(all_df_test, X_test, y_test_pair, model, threshold=0.5):
    
    #Convert pair-wise predictions to event-wise decisions

    #Parameters:
    #- all_df_test: Dataframe with event information (contains 'event' columns)
    #- X_test: Feature matrix for pairs
    #- y_test_pair: True labels for pairs (1 for pi0, 0 for not pi0)
    #- threshold: Probability threshold for pair classification

    #Returns:
    #- event_results: DataFrame with event-wise predictions and truths

    # Debug: Print shapes to identify the issue
    print(f"Debug - all_df_test shape: {all_df_test.shape}")
    print(f"Debug - X_test shape: {X_test.shape}")
    print(f"Debug - y_test_pair shape: {y_test_pair.shape}")
    print(f"Debug - Number of unique events in all_df_test: {all_df_test['event'].nunique()}")

    # Get photon counts per event
    event_photon_counts = all_df_test.groupby('event').size().to_dict()
    print(f"Debug - Photon counts per event (sample): {dict(list(event_photon_counts.items())[:5])}")
    
    # Calculate expected number of pairs per event (all combinations)
    event_expected_pairs = {}
    for event_id, n_photons in event_photon_counts.items():
        event_expected_pairs[event_id] = n_photons * (n_photons - 1) // 2
    
    total_expected_pairs = sum(event_expected_pairs.values())
    print(f"Debug - Total expected pairs (all combinations): {total_expected_pairs}")
    print(f"Debug - Actual pairs in X_test: {len(X_test)}")

    # Get pair preditions from model
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of being pi0
    y_pred_pair = (y_pred_proba >= threshold).astype(int) 

    print(f"Debug - y_pred_proba shape: {y_pred_proba.shape}")

    # Get the event for each photon (each photon appears in multiple pairs)
    photon_events = all_df_test['event'].values

    # Let's reconstruct by assuming pairs are grouped by event
    unique_events = sorted(all_df_test['event'].unique())

    # Reconstruct event IDs for each pair
    pair_event_ids = []
    
    for event_id in unique_events:
        n_photons = event_photon_counts[event_id]
        
        # Calculate number of pairs for this event
        # For an event with N photons, there are N*(N-1)/2 possible pairs
        n_pairs_in_event = n_photons * (n_photons - 1) // 2
        
        # Add event ID for each pair in this event
        pair_event_ids.extend([event_id] * n_pairs_in_event)
    
    print(f"Debug - Reconstructed {len(pair_event_ids)} pair event IDs")
    
    # Check if we have the right number of pairs (should be 3x photons)
    expected_pairs = len(all_df_test) * 3  # Based on the ratio you observed
    print(f"Debug - Expected pairs (3x photons): {expected_pairs}")

    if len(pair_event_ids) != len(y_pred_proba):
        print(f"Warning: Length mismatch! Reconstructed {len(pair_event_ids)} but need {len(y_pred_proba)}")
        
    # If we have too many, truncate
    if len(pair_event_ids) > len(y_pred_proba):
        print(f"Truncating to {len(y_pred_proba)} pairs")
        pair_event_ids = pair_event_ids[:len(y_pred_proba)]
    else:
        # If we have too few, we need to pad
        print(f"WARNING: Need to pad {len(y_pred_proba) - len(pair_event_ids)} pairs")
        # This shouldn't happen if our assumption is correct
        # Let's repeat the last event ID to fill
        last_event_id = pair_event_ids[-1] if pair_event_ids else 0
        pair_event_ids.extend([last_event_id] * (len(y_pred_proba) - len(pair_event_ids)))

    # Create a DataFrame with pair information
    pair_df = pd.DataFrame({
        'event': all_df_test['event'].values, # Event ID
        'pair_index': range(len(y_test_pair)), # Pair index within event
        'true_pair': y_test_pair.values, # True label for this pair
        'pred_pair': y_pred_pair, # Predicted label for this pair
        'pred_proba': y_pred_proba # Prediction probability
    })

    # Group by event to make event-wise decisions
    event_results = []

    for event_id, group in pair_df.groupby('event'):
        # Get true signal status for this event from all_df_test
        # Assuming 'is_signal' column exists in all_df_test (1 for signal events, 0 for background)
        event_true = all_df_test[all_df_test['event'] == event_id]['is_signal'].iloc[0]

        # Event-wise prediction logic:
        # Option 1: Event is signal if at least one pair is predicted as pi0
        #event_pred_any = int(group['pred_pair'].sum() > 0)

        # Option 2: Event is signal if at least N pairs are pedicted as pi0
        #event_pred_min = int(group['pred_pair'].sum() >= 2)

        # Option 3: Event is signal is max prediction probability > threshold
        event_pred_max = int(group['pred_proba'].max() >= threshold)

        # Option 4: Event is signal if average prediction > threshold
        #event_pred_mean = int(group['pred_proba'].mean() >= threshold)

        event_results.append({
            'event_id': event_id,
            'true_signal': event_true,
            'pred_signal': event_pred_max, # Using "max" strategy
            'n_pairs': len(group),
            'n_pred_pi0': group['pred_pair'].sum(),
            'max_proba': group['pred_proba'].max(),
            'mean_proba': group['pred_proba'].mean()
        })

    return pd.DataFrame(event_results)
'''

def plot_event_confusion_matrix():
    """
    Plot confusion matrix for event-wise classification
    """

    print("Plot confusion matrix for event-wise classification")

def analyze_threshold_impact():
    """
    Analyze how different thresholds affect event-wise classification
    """

    print("Analyze how different thresholds affect event-wise classification")

if __name__ == '__main__':

    print(f"Application on test dataset...")
    
    input_data_dir = os.path.join(project_root, f'analysis/dataset')
    input_model_dir = os.path.join(project_root, f'training/models')

    category_type = 'TCOMB' #'TCOMB' #'TETAGAM', 'TISR3PI_SIG', 'TKSL'

    phys_map = joblib.load(os.path.join(input_data_dir, f'phys_map.pkl'))
    print(phys_map)
    

    # Create output folder
    plot_dir = os.path.join(project_root, 'application/plots')

    # With this (always fresh) plot_dir
    import shutil
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    
    for data_type, info in phys_map.items():
        br_title = info['br_title']
        category = info['category']
        #print(f"Inspecting dataset {data_type}; {br_nm}; {br_title}; {category}")  
        print(info)

        if (data_type == category_type):

            # Load test dataset and all_df
            all_df, all_df_test, model, X_test, y_test = load_data_model(data_type)

            # Selection cut (chi2, E_dela, opening angle, beta)

            # Check kine

            ## Plot confusion matrix (event-basis)
            
            # Get event-wise prediction
            #event_results = event_wise_prediction(
            #    all_df_test, X_test, y_test, model, threshold=0.5
            #)

            '''
            all_df_test contains the event column that links pairs to events
            The is_signal column in all_df_test tells you which events are truly signal
            We need an aggreation strategy to convert multiple pair predictions into one event dicision:
                "Any" strategy: Event is signal if at least one pair is predicted as pi0
                "Min count" stragegy: Require at least N pairs predicted as pi0
                "Max probability" strategy: Use the maximum prediction probality
                "Mean probability" strategy: Average all pair probabilty
            The confusion matrix will show:
                True Positives: Signal events correctly identified
                True Negatives: Background events correctly identifed
                False Positives: Background events misidentified as signal
                False Negatives: Signal events misidentified as background
            '''

            
            ## Plot confusion matrix (photon features)
            fig_cm = plot_nm(X_test, y_test, model, br_title)
            fig_cm.savefig(f'{plot_dir}/cm_{data_type}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_cm)

            ## Accuracy metrics, event basis
            score_list, var_list, var_str = event_performance(all_df, model)

            fig_var_score = plot_var_score(var_list, score_list, var_str, f"Mass and Score (test, {br_title})")
            fig_var_score.savefig(f'{plot_dir}/pi0_mass_score_{data_type}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_var_score)

            ## ROC plot
            fig_roc = plot_roc(score_list, rf'ROC Curve - $\pi^{0}$ Classifier (test, {br_title})')
            fig_roc.savefig(f'{plot_dir}/roc_curv_{data_type}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_roc)

            ## Plot kine. var after the pi0 identification
        

        else:
            print("No true labels")

   