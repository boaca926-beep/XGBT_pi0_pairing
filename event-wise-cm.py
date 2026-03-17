Key Points for Event-wise Evaluation:

    Your all_df_test contains the event column that links pairs to events

    The is_signal column in all_df_test tells you which events are truly signal

    You need an aggregation strategy to convert multiple pair predictions into one event decision:

        "Any" strategy: Event is signal if at least one pair is predicted as pi0

        "Min count" strategy: Require at least N pairs predicted as pi0

        "Max probability" strategy: Use the maximum prediction probability

        "Mean probability" strategy: Average all pair probabilities

    The confusion matrix will show:

        True Positives: Signal events correctly identified

        True Negatives: Background events correctly identified

        False Positives: Background events misidentified as signal

        False Negatives: Signal events misidentified as background
        
        
=====================================================================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

def event_wise_prediction(all_df_test, X_test, y_test_pair, model, threshold=0.5):
    """
    Convert pair-wise predictions to event-wise decisions
    
    Parameters:
    - all_df_test: DataFrame with event information (contains 'event' column)
    - X_test: Feature matrix for pairs
    - y_test_pair: True labels for pairs (1 for pi0, 0 for not pi0)
    - model: Trained classifier
    - threshold: Probability threshold for pair classification
    
    Returns:
    - event_results: DataFrame with event-wise predictions and truths
    """
    
    # Get pair predictions from model
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of being pi0
    y_pred_pair = (y_pred_proba >= threshold).astype(int)
    
    # Create a DataFrame with pair information
    pair_df = pd.DataFrame({
        'event': all_df_test['event'].values,  # Event ID
        'pair_index': range(len(y_test_pair)),  # Pair index within event
        'true_pair': y_test_pair.values,  # True label for this pair
        'pred_pair': y_pred_pair,  # Predicted label for this pair
        'pred_proba': y_pred_proba  # Prediction probability
    })
    
    # Group by event to make event-wise decisions
    event_results = []
    
    for event_id, group in pair_df.groupby('event'):
        # Get true signal status for this event from all_df_test
        # Assuming 'is_signal' column exists in all_df_test (1 for signal events, 0 for background)
        event_true = all_df_test[all_df_test['event'] == event_id]['is_signal'].iloc[0]
        
        # Event-wise prediction logic:
        # Option 1: Event is signal if at least one pair is predicted as pi0
        event_pred_any = int(group['pred_pair'].sum() > 0)
        
        # Option 2: Event is signal if at least N pairs are predicted as pi0
        # event_pred_min = int(group['pred_pair'].sum() >= 2)
        
        # Option 3: Event is signal if max prediction probability > threshold
        # event_pred_max = int(group['pred_proba'].max() >= threshold)
        
        # Option 4: Event is signal if average prediction > threshold
        # event_pred_mean = int(group['pred_proba'].mean() >= threshold)
        
        event_results.append({
            'event_id': event_id,
            'true_signal': event_true,
            'pred_signal': event_pred_any,  # Using "any" strategy
            'n_pairs': len(group),
            'n_pred_pi0': group['pred_pair'].sum(),
            'max_proba': group['pred_proba'].max(),
            'mean_proba': group['pred_proba'].mean()
        })
    
    return pd.DataFrame(event_results)

def plot_event_confusion_matrix(event_results, save_path=None):
    """
    Plot confusion matrix for event-wise classification
    """
    # Create confusion matrix
    cm = confusion_matrix(
        event_results['true_signal'], 
        event_results['pred_signal']
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Background', 'Signal']
    )
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    # Customize
    plt.title('Event-wise Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    tn, fp, fn, tp = cm.ravel()
    
    # Add text with percentages
    ax.text(0.3, 0.3, f'{tn} ({tn/total*100:.1f}%)', 
            transform=ax.transAxes, fontsize=12, ha='center')
    ax.text(0.7, 0.3, f'{fp} ({fp/total*100:.1f}%)', 
            transform=ax.transAxes, fontsize=12, ha='center')
    ax.text(0.3, 0.7, f'{fn} ({fn/total*100:.1f}%)', 
            transform=ax.transAxes, fontsize=12, ha='center')
    ax.text(0.7, 0.7, f'{tp} ({tp/total*100:.1f}%)', 
            transform=ax.transAxes, fontsize=12, ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print metrics
    print("\n" + "="*50)
    print("EVENT-WISE CLASSIFICATION METRICS")
    print("="*50)
    print(f"Total events: {total}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn} ({tn/total*100:.1f}%)")
    print(f"  False Positives: {fp} ({fp/total*100:.1f}%)")
    print(f"  False Negatives: {fn} ({fn/total*100:.1f}%)")
    print(f"  True Positives:  {tp} ({tp/total*100:.1f}%)")
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:    {accuracy:.3f}")
    print(f"  Precision:   {precision:.3f}")
    print(f"  Recall:      {recall:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  F1-Score:    {f1:.3f}")

def analyze_threshold_impact(all_df_test, X_test, y_test_pair, model):
    """
    Analyze how different thresholds affect event-wise classification
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []
    
    for thresh in thresholds:
        event_results = event_wise_prediction(
            all_df_test, X_test, y_test_pair, model, threshold=thresh
        )
        
        tn, fp, fn, tp = confusion_matrix(
            event_results['true_signal'], 
            event_results['pred_signal']
        ).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
    
    # Plot threshold impact
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Metrics vs threshold
    ax = axes[0]
    ax.plot(results_df['threshold'], results_df['accuracy'], 'o-', label='Accuracy', linewidth=2)
    ax.plot(results_df['threshold'], results_df['precision'], 's-', label='Precision', linewidth=2)
    ax.plot(results_df['threshold'], results_df['recall'], '^-', label='Recall', linewidth=2)
    ax.set_xlabel('Probability Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Event-wise Performance vs Threshold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Confusion matrix counts vs threshold
    ax = axes[1]
    ax.plot(results_df['threshold'], results_df['tp'], 'g-', label='True Positives', linewidth=2)
    ax.plot(results_df['threshold'], results_df['tn'], 'b-', label='True Negatives', linewidth=2)
    ax.plot(results_df['threshold'], results_df['fp'], 'r-', label='False Positives', linewidth=2)
    ax.plot(results_df['threshold'], results_df['fn'], 'orange', label='False Negatives', linewidth=2)
    ax.set_xlabel('Probability Threshold', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Event Counts vs Threshold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# ============================================================================
# MAIN USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    # Load your trained model and data
    # model = joblib.load('path/to/your/model.pkl')
    
    # Load test data (assuming you have a trained model)
    data_dir = './dataset'
    
    # Load the event-level DataFrame for test set
    all_df_test = joblib.load(f'{data_dir}/all_df_test_TCOMB.pkl')
    
    # Load pair-level features and labels
    X_test = joblib.load(f'{data_dir}/X_test_TCOMB.pkl')
    y_test = joblib.load(f'{data_dir}/y_test_TCOMB.pkl')  # These are pair labels
    
    print(f"Event-level test data: {len(all_df_test)} events")
    print(f"Pair-level test data: {len(X_test)} pairs")
    print(f"Average pairs per event: {len(X_test)/len(all_df_test):.1f}")
    
    # Example: If you have a trained model
    # from your_training_script import load_model
    # model = load_model()
    
    # For demonstration, let's assume you have a model
    # If not, you'll need to train one first
    
    '''
    # After training your model:
    
    # Get event-wise predictions
    event_results = event_wise_prediction(
        all_df_test, X_test, y_test, model, threshold=0.5
    )
    
    # Plot event-wise confusion matrix
    plot_event_confusion_matrix(
        event_results, 
        save_path='./plots/event_confusion_matrix.png'
    )
    
    # Analyze threshold impact
    threshold_analysis = analyze_threshold_impact(
        all_df_test, X_test, y_test, model
    )
    
    # Find optimal threshold (e.g., maximizing F1 or accuracy)
    optimal_idx = threshold_analysis['accuracy'].idxmax()
    optimal_thresh = threshold_analysis.loc[optimal_idx, 'threshold']
    print(f"\nOptimal threshold for accuracy: {optimal_thresh:.2f}")
    '''
