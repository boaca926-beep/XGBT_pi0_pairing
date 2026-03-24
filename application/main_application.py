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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from config import DATA_DIR, DATA_LARGE_DIR, PLOT_DIR_VAL, PLOT_DIR_APP, MODEL_DIR


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


def event_wise_prediction(all_df_test, X_test, y_test_pair, model, threshold=0.5):
    
    '''
    Convert pair-wise predictions to event-wise decisions

    Parameters:
    - all_df_test: Dataframe with event information (contains 'event' columns)
    - X_test: Feature matrix for pairs
    - y_test_pair: True labels for pairs (1 for pi0, 0 for not pi0)
    - threshold: Probability threshold for pair classification

    Returns:
    - event_results: DataFrame with event-wise predictions and truths

    Key insight:
    - Signal events: π⁰ → γγ, so exactly 2 photons per signal event
    - Background events: Mostly single photons
    - Pairs are constructed per photon (3 nearest neighbors)
    - We need to determine if an event contains a π⁰ decay
    
    Strategy: An event is signal if its two photons are both identified
    as part of π⁰ candidates (connected through pair predictions)
    '''

    # Debug: Print shapes to identify the issue
    print(f"Debug - all_df_test shape: {all_df_test.shape}")
    print(f"Debug - X_test shape: {X_test.shape}")
    print(f"Debug - y_test_pair shape: {y_test_pair.shape}")
    print(f"Debug - Number of unique events in all_df_test: {all_df_test['event'].nunique()}")

    # Get pair preditions from model
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of being pi0
    y_pred_pair = (y_pred_proba >= threshold).astype(int) 
    print(f"Debug - y_pred_proba shape: {y_pred_proba.shape}")

    # ============ UNDERSTAND THE TRUE LABELS ============
    print("\n" + "="*60)
    print("UNDERSTANDING TRUE LABELS")
    print("="*60)
    
    # Check what columns are available in all_df_test
    print(f"\nColumns in all_df_test: {list(all_df_test.columns)}")
    
    # Check the distribution of is_signal_photon per event
    print(f"\nDistribution of is_signal_photon per event:")
    signal_per_event = all_df_test.groupby('event')['is_signal'].agg(['sum', 'count']).reset_index()
    signal_per_event.columns = ['event', 'n_signal_photons', 'n_total_photons']
    
    print(f"  Events with 0 signal photons: {(signal_per_event['n_signal_photons'] == 0).sum()}")
    print(f"  Events with 1 signal photon: {(signal_per_event['n_signal_photons'] == 1).sum()}")
    print(f"  Events with 2 signal photons: {(signal_per_event['n_signal_photons'] == 2).sum()}")
    print(f"  Events with >2 signal photons: {(signal_per_event['n_signal_photons'] > 2).sum()}")
    
    # Check if there's an event-level signal flag
    if 'is_signal_event' in all_df_test.columns:
        print(f"\nEvent-level signal flag found!")
        event_level_signal = all_df_test.groupby('event')['is_signal_event'].first()
        print(f"  Signal events: {event_level_signal.sum()}")
        print(f"  Background events: {len(event_level_signal) - event_level_signal.sum()}")
    else:
        print(f"\nNo event-level signal flag found. Using event with 2 signal photons as signal.")
    
    # Check what the pair-level labels represent
    print(f"\nPair-level labels (y_test_pair):")
    print(f"  π⁰ pairs: {(y_test_pair == 1).sum()}")
    print(f"  Non-π⁰ pairs: {(y_test_pair == 0).sum()}")
    
    # ============ PHOTON-LEVEL AGGREGATION ============
    n_photons = len(all_df_test)
    pairs_per_photon = 3
    
    print(f"\nCreating photon-level aggregation...")
    print(f"  Total photons: {n_photons}")
    print(f"  Pairs per photon: {pairs_per_photon}")
    
    # Aggregate predictions at photon level
    photon_data = []
    
    for photon_idx in range(n_photons):
        start_idx = photon_idx * pairs_per_photon
        end_idx = start_idx + pairs_per_photon
        
        if end_idx > len(y_pred_proba):
            end_idx = len(y_pred_proba)
            if start_idx >= end_idx:
                continue
        
        event_id = all_df_test.iloc[photon_idx]['event']
        is_signal_photon = all_df_test.iloc[photon_idx].get('is_signal', 0)
        
        photon_probs = y_pred_proba[start_idx:end_idx]
        photon_preds = y_pred_pair[start_idx:end_idx]
        
        # Check if this photon is part of a true π⁰ pair
        has_true_pi0 = False
        for pair_idx in range(start_idx, end_idx):
            if pair_idx < len(y_test_pair) and y_test_pair.iloc[pair_idx] == 1:
                has_true_pi0 = True
                break
        
        photon_data.append({
            'photon_idx': photon_idx,
            'event_id': event_id,
            'is_signal_photon': is_signal_photon,
            'has_true_pi0': has_true_pi0,
            'max_proba': photon_probs.max(),
            'mean_proba': photon_probs.mean(),
            'n_pi0_pairs': photon_preds.sum(),
            'has_pi0_pair': int(photon_preds.sum() > 0),
        })
    
    photon_df = pd.DataFrame(photon_data)
    
    # ============ EVENT-LEVEL AGGREGATION ============
    # Let's define signal events based on the pair-level truth
    # An event is truly signal if it contains at least one true π⁰ pair
    # (since π⁰ → γγ, a true π⁰ pair connects two photons in the same event)
    
    event_results = []
    
    for event_id, group in photon_df.groupby('event_id'):
        n_photons_in_event = len(group)
        
        # TRUE LABEL: Event is signal if it contains at least one true π⁰ pair
        true_signal = int(group['has_true_pi0'].sum() > 0)
        
        # Alternative: Use photon-level signal flag (if available)
        true_signal_photon_based = int(group['is_signal_photon'].sum() >= 2)  # At least 2 signal photons
        
        # PREDICTION: Event is signal if at least one photon has a predicted π⁰ pair
        pred_any = int(group['has_pi0_pair'].sum() > 0)
        
        # Prediction: Event is signal if at least 2 photons have predicted π⁰ pairs
        pred_min2 = int(group['has_pi0_pair'].sum() >= 2)
        
        # Prediction: Event is signal if max probability > threshold
        pred_max = int(group['max_proba'].max() >= threshold)
        
        event_results.append({
            'event_id': event_id,
            'true_signal_pair_based': true_signal,
            'true_signal_photon_based': true_signal_photon_based,
            'n_photons': n_photons_in_event,
            'n_signal_photons': group['is_signal_photon'].sum(),
            'n_photons_with_true_pi0': group['has_true_pi0'].sum(),
            'n_photons_with_pred_pi0': group['has_pi0_pair'].sum(),
            'max_proba': group['max_proba'].max(),
            'pred_any': pred_any,
            'pred_min2': pred_min2,
            'pred_max': pred_max
        })
    
    event_df = pd.DataFrame(event_results)
    
    # ============ EVALUATION ============
    print("\n" + "="*60)
    print("EVENT-LEVEL ANALYSIS")
    print("="*60)
    
    print(f"\nTrue event distribution (based on π⁰ pairs):")
    true_signal_count = event_df['true_signal_pair_based'].sum()
    print(f"  Signal events: {true_signal_count}")
    print(f"  Background events: {len(event_df) - true_signal_count}")
    
    print(f"\nTrue event distribution (based on photon signal flag >=2):")
    true_signal_photon_count = event_df['true_signal_photon_based'].sum()
    print(f"  Signal events: {true_signal_photon_count}")
    
    # Check alignment between the two definitions
    aligned = ((event_df['true_signal_pair_based'] == 1) & (event_df['true_signal_photon_based'] == 1)).sum()
    print(f"\nEvents that are signal by both definitions: {aligned}")
    
    print(f"\nPrediction on all events:")
    print(f"  Strategy 'any': {event_df['pred_any'].sum()} events predicted as signal")
    print(f"  Strategy 'min2': {event_df['pred_min2'].sum()} events predicted as signal")
    print(f"  Strategy 'max': {event_df['pred_max'].sum()} events predicted as signal")
    
    # Calculate metrics using the correct true label (based on π⁰ pairs)
    strategies = [('any', 'pred_any'), ('min2', 'pred_min2'), ('max', 'pred_max')]
    best_f1 = 0
    best_strategy = 'any'
    
    for strategy_name, col in strategies:
        tp = ((event_df['true_signal_pair_based'] == 1) & (event_df[col] == 1)).sum()
        fp = ((event_df['true_signal_pair_based'] == 0) & (event_df[col] == 1)).sum()
        tn = ((event_df['true_signal_pair_based'] == 0) & (event_df[col] == 0)).sum()
        fn = ((event_df['true_signal_pair_based'] == 1) & (event_df[col] == 0)).sum()
        
        if (tp + fp + tn + fn) > 0:
            accuracy = (tp + tn) / len(event_df)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nStrategy '{strategy_name}':")
            print(f"  TP: {tp:5d} | FP: {fp:5d} | TN: {tn:5d} | FN: {fn:5d}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_strategy = strategy_name
        else:
            print(f"\nStrategy '{strategy_name}': No events to evaluate")
    
    print(f"\nBest strategy: '{best_strategy}' with F1 = {best_f1:.4f}")
    
    # Add the best strategy as default
    event_df['pred_signal'] = event_df[f'pred_{best_strategy}']
    
    return event_df
    

def plot_event_confusion_matrix(event_results, data_type, plot_dir):
    """
    Plot confusion matrix for event-wise classification
    """
 
    
    # Calculate confusion matrix
    cm = confusion_matrix(event_results['true_signal_pair_based'], 
                          event_results['pred_signal'])
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Background', 'Signal'],
                yticklabels=['Background', 'Signal'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'Confusion Matrix (Counts) - {data_type}')
    
    # Plot percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=axes[1],
                xticklabels=['Background', 'Signal'],
                yticklabels=['Background', 'Signal'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title(f'Confusion Matrix (Percentages) - {data_type}')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/event_cm_{data_type}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Calculate and print metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nEvent-wise Classification Metrics:")
    print(f"  True Positives:  {tp:5d}")
    print(f"  False Positives: {fp:5d}")
    print(f"  True Negatives:  {tn:5d}")
    print(f"  False Negatives: {fn:5d}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return fig

def analyze_threshold_impact(event_results, data_type, plot_dir):
    """
    Analyze how different thresholds affect event-wise classification
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []
    
    for threshold in thresholds:
        # Re-calculate predictions with new threshold using 'any' strategy
        # We need to recompute photon-level predictions at this threshold
        # For simplicity, we'll use the stored max_proba values
        pred_signal = (event_results['max_proba'] >= threshold).astype(int)
        
        tp = ((event_results['true_signal_pair_based'] == 1) & (pred_signal == 1)).sum()
        fp = ((event_results['true_signal_pair_based'] == 0) & (pred_signal == 1)).sum()
        tn = ((event_results['true_signal_pair_based'] == 0) & (pred_signal == 0)).sum()
        fn = ((event_results['true_signal_pair_based'] == 1) & (pred_signal == 0)).sum()
        
        accuracy = (tp + tn) / len(event_results) if len(event_results) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    results_df = pd.DataFrame(results)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(results_df['threshold'], results_df['accuracy'], 'b-', linewidth=2, label='Accuracy')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(results_df['threshold'], results_df['precision'], 'r-', linewidth=2, label='Precision')
    axes[0, 1].plot(results_df['threshold'], results_df['recall'], 'g-', linewidth=2, label='Recall')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(results_df['threshold'], results_df['f1'], 'm-', linewidth=2, label='F1 Score')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Find best threshold
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_f1 = results_df.loc[best_idx, 'f1']
    best_precision = results_df.loc[best_idx, 'precision']
    best_recall = results_df.loc[best_idx, 'recall']
    
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.7, f'Best Threshold Analysis:', fontsize=12, fontweight='bold')
    axes[1, 1].text(0.1, 0.5, f'Best Threshold: {best_threshold:.3f}', fontsize=11)
    axes[1, 1].text(0.1, 0.4, f'F1 Score: {best_f1:.4f}', fontsize=11)
    axes[1, 1].text(0.1, 0.3, f'Precision: {best_precision:.4f}', fontsize=11)
    axes[1, 1].text(0.1, 0.2, f'Recall: {best_recall:.4f}', fontsize=11)
    
    plt.suptitle(f'Threshold Impact Analysis - {data_type}')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/threshold_analysis_{data_type}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nBest threshold: {best_threshold:.3f} (F1 = {best_f1:.4f})")
    
    return results_df, best_threshold

if __name__ == '__main__':

    print(f"Application on test dataset...")
    
    #input_data_dir = os.path.join(project_root, f'analysis/dataset')
    #input_model_dir = os.path.join(project_root, f'training/models')
    input_data_dir = DATA_DIR #DATA_LARGE_DIR
    input_model_dir = MODEL_DIR

    category_type = 'TETAGAM' #'TCOMB' #'TETAGAM', 'TISR3PI_SIG', 'TKSL'

    phys_map = joblib.load(os.path.join(input_data_dir, f'phys_map.pkl'))
    print(phys_map)

    # Create output folder
    plot_dir = PLOT_DIR_APP #os.path.join(project_root, 'application/plots')

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
            event_results = event_wise_prediction(
                all_df_test, X_test, y_test, model, threshold=0.5
            )

            # Plot event confusion matrix
            plot_event_confusion_matrix(event_results, data_type, plot_dir)
            
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

            # Analyze threshold impact
            threshold_results, best_threshold = analyze_threshold_impact(
                event_results, data_type, plot_dir
            )

            ## Plot confusion matrix (photon features)
            fig_cm = plot_nm(X_test, y_test, model, br_title)
            fig_cm.savefig(f'{plot_dir}/cm_{data_type}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_cm)

            # Accuracy metrics, event basis
            score_list, var_list, var_str = event_performance(all_df, model)

            fig_var_score = plot_var_score(var_list, score_list, var_str, f"Mass and Score (test, {br_title})")
            fig_var_score.savefig(f'{plot_dir}/pi0_mass_score_{data_type}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_var_score)

            # ROC plot
            fig_roc = plot_roc(score_list, rf'ROC Curve - $\pi^{0}$ Classifier (test, {br_title})')
            fig_roc.savefig(f'{plot_dir}/roc_curv_{data_type}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_roc)
            
            # Save event results
            event_results.to_csv(f'{plot_dir}/event_results_{data_type}.csv', index=False)

            # Save summary
            with open(f'{plot_dir}/summary_{data_type}.txt', 'w') as f:
                f.write(f"Dataset: {data_type}\n")
                f.write(f"Title: {br_title}\n")
                f.write(f"Category: {category}\n")
                f.write(f"\nEvent-wise Classification Summary:\n")
                f.write(f"Best Strategy: 'any' (event is signal if any photon has π⁰ pair)\n")
                f.write(f"Best Threshold: {best_threshold:.3f}\n")
                f.write(f"Total events: {len(event_results)}\n")
                f.write(f"Signal events: {event_results['true_signal_pair_based'].sum()}\n")
                f.write(f"Background events: {len(event_results) - event_results['true_signal_pair_based'].sum()}\n")
                
                # Calculate final metrics
                tp = ((event_results['true_signal_pair_based'] == 1) & (event_results['pred_signal'] == 1)).sum()
                fp = ((event_results['true_signal_pair_based'] == 0) & (event_results['pred_signal'] == 1)).sum()
                tn = ((event_results['true_signal_pair_based'] == 0) & (event_results['pred_signal'] == 0)).sum()
                fn = ((event_results['true_signal_pair_based'] == 1) & (event_results['pred_signal'] == 0)).sum()
                
                accuracy = (tp + tn) / len(event_results)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write(f"\nFinal Metrics (threshold=0.5):\n")
                f.write(f"  TP: {tp}\n")
                f.write(f"  FP: {fp}\n")
                f.write(f"  TN: {tn}\n")
                f.write(f"  FN: {fn}\n")
                f.write(f"  Accuracy: {accuracy:.4f}\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall: {recall:.4f}\n")
                f.write(f"  F1 Score: {f1:.4f}\n")
            
            print(f"\n✓ All results saved to {plot_dir}")

            ## Accuracy metrics, event basis
            #score_list, var_list, var_str = event_performance(all_df, model)

            #fig_var_score = plot_var_score(var_list, score_list, var_str, f"Mass and Score (test, {br_title})")
            #fig_var_score.savefig(f'{plot_dir}/pi0_mass_score_{data_type}.png', dpi=300, bbox_inches='tight')
            #plt.close(fig_var_score)

            ## ROC plot
            #fig_roc = plot_roc(score_list, rf'ROC Curve - $\pi^{0}$ Classifier (test, {br_title})')
            #fig_roc.savefig(f'{plot_dir}/roc_curv_{data_type}.png', dpi=300, bbox_inches='tight')
            #plt.close(fig_roc)

            ## Plot kine. var after the pi0 identification
        

        else:
            print("No true labels")

   