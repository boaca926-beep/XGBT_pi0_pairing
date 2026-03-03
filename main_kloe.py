import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
import warnings
from methods import prepare_3photon_paris, plot_hist, find_best_pi0_candidate, set_model_params, MC_generation
from methods import kloe_sample, plot_compr_hist, plot_feature_pairs, plot_feature_target, plot_var
from methods import plot_var_score, plot_roc, plot_learning_curves, baye_opti
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
import joblib
import os


if __name__ == "__main__":

    # Use KLOE dataset
    #input_str = ['isr3pi', 'TISR3PI_SIG'] # Stored root file type [0], and branch name [1] 
    input_str = ['etagam', 'TETAGAM']

    # physical channels
    phys_map = {
    'TETAGAM': rf"$e^{{+}}e^{{-}}\to\phi\to\eta\gamma$", 
    'TISR3PI_SIG': rf"$e^{{+}}e^{{-}}\to\pi^{{+}}\pi^{{-}}\pi^{{0}}\gamma$" 
    }

    global phys_ch
    phys_ch = phys_map.get(input_str[1], "")
    print(f"Physical channel {phys_ch}, branch {input_str[1]}")

    all_df, good_df, bad_df = kloe_sample(input_str)
    #print(good_E1.head(5))
    print(f'len bad     ', (all_df['true_pi0_pair'] == (-1, -1)).sum())
    print(all_df['true_pi0_pair'].value_counts(), f'len good    {len(good_df)}')

    #nb_all_df = [i for i in range(len(all_df))] # Vector of number of all signal events
    #all_df['event'] = nb_all_df
    #print(f'all_df\n    ', all_df.head(6))
    #print(f'good_df\n   ', good_df.head(6))
    #print(f'bad_df\n   ', bad_df.head(6))

    #if True:
    #    sys.exit("Exiting program")  # Program ends here

    #columns = [col for col in all_df.columns]
    all_df_plot = all_df.drop(['event', 'is_signal', 'true_pi0_pair'], axis=1) # Ready for plot
    good_df_plot = good_df.drop(['event'], axis=1)
    bad_df_plot = bad_df.drop(['event'], axis=1)
    #print(all_df_plot.head(8))
    #print(good_df.head(8))

    #plot_hist(all_df_plot, 3, 100, "Photon 4-momentum (all signal)") # data, column list, number of rows to plot, number of bins
    #plot_hist(good_df_plot, 3, 100, "Photon 4-momentum (good signal)") 
    #plot_hist(bad_df_plot, 3, 100, "Photon 4-momentum (bad signal)") 

    df_set = [all_df_plot, good_df_plot, bad_df_plot]
    #plot_compr_hist(df_set, 3, 100, rf"$\gamma$ 4-momentum ({phys_ch})", "Photon 4-momentum") # Photon 4-momentum comparison plot

    ## Prepare pair dataset with EXACT 4-vector quantities
    pair_all_df = prepare_3photon_paris(all_df)
    #print(pair_all_df.describe())
    #print(pair_all_df.head(5))
    pair_good_df = pair_all_df[pair_all_df['is_pi0'] == 1]
    pair_bad_df = pair_all_df[pair_all_df['is_pi0'] == 0]
    #print(pair_good_df.head(5))
    print(f"total pi0 candidates    {len(pair_all_df)}\ngood candidates     {len(pair_good_df)}\nbad candidates     {len(pair_bad_df)}")
    print(f"good ratio  {len(pair_good_df)/len(pair_all_df):.2f}")

    ## Check photon pair quantites
    #pos_pi0_mass = pair_df[pair_all_df['is_pi0'] == 1]['m_gg'].tolist()
    pos_pi0_mass = pair_good_df['m_gg'].tolist()
    #pos_pi0_mass = pair_bad_df['m_gg'].tolist()
    #pos_pi0_mass = pair_all_df['m_gg'].tolist()

    #plot_var(pos_pi0_mass, 'm_gg', phys_ch)

    ## Plot comparsion of paired photon quantities
    drop_columns = ['event', 'pair_id', 'is_pi0']
    pair_df_plot = pair_all_df.drop(drop_columns, axis=1) # drop is_pi0, prepare for plot
    pair_good_plot = pair_good_df.drop(drop_columns, axis=1)
    pair_bad_plot = pair_bad_df.drop(drop_columns, axis=1)

    plot_df_set = [pair_df_plot, pair_good_plot, pair_bad_plot]
    #plot_compr_hist(plot_df_set, 2, 100, rf"$\pi^{0}$ Candidates ({phys_ch})", rf"pi0_compr") # Pi0 comparison plot
    
    #print(pair_all_df.columns) 

    ## Feature-feature correlations
    #plot_feature_pairs(pair_all_df, rf"$\pi^{0}$ Candidates Feature-feature (Signal=Blue, Background=Red) ({phys_ch})", 'feature-feautre_correlation')

    ## Feature-target correlations
    feature_columns = ['m_gg', 'opening_angle', 'cos_theta', 'E_asym', 'e_min_x_angle', 'E1', 'E2', 'E_ratio', 'is_pi0']
    target_corr = pair_all_df[feature_columns].corr()['is_pi0'].drop('is_pi0') #.sort_values(ascending=False)
    sorted_by_abs = target_corr.abs().sort_values(ascending=False)
    #print(f"target_corr type: {type(target_corr), {target_corr.shape}}")
    #print(sorted_by_abs)
    #print(pair_all_df[['E_asym']].describe())

    #plot_feature_target(target_corr, rf'Feature Importance: Correlation with true $\pi^{0}$ ({phys_ch})', 'feature_target_correlation')

    ## Bayesian optimization (look for best model parameters)
    
    features = ['m_gg', 'opening_angle', 'cos_theta', 'E_asym'] # Features: EXACT physics quantities from 4-vectors
    X = pair_all_df[features]
    y = pair_all_df['is_pi0']

    #print(y[y == 1], "y: ", type(y), "n_ones = ", sum(y == 1)) # print only signal events
    #print(X, "X: ", type(X))
    #print(f"pair_df:    {pair_df.iloc[:, 2]}")

    ## Split (train + test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42
    )

    # Further split training into (train + val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42  # 0.25 of 80% = 20% of total
    )

    #   Training: 60% of data (0.8 × 0.75 = 0.6)
    #   Validation: 20% of data (0.8 × 0.25 = 0.2)
    #   Test: 20% of data (the original 0.2)

    ## Split train
    #X_train, X_val, y_train, y_val = train_test_split(
    #    X, y, test_size = 0.2, random_state = 42 # training (80%) and validation (20%) sets.
    #)
 
    ## Train classifier (train with optimized model parameters)
    print("\n* Training XGBoost on EXACT 4-vector features ...")
    params = baye_opti(X_train, y_train) # Find best model parameters
    #params = set_model_params(X_train, y_train) # Initialize model parameters
    params['early_stopping_rounds'] = 50 # Add early stop parameter
    # print("\nModel parameters:", params)

    model = xgb.XGBClassifier(**params) # Create a model, Fit with the model, and save
    model.fit(X_train, y_train,
              eval_set = [(X_train, y_train), (X_val, y_val)],
              verbose=False
    )
    model_path = './models/pi0_classifier_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    ## Evaluate validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)

    print(f"\nValidation Performance:")
    print(f"AUC: {auc:.3f}")
    print(f"Accuracy: {acc:.4f}")

    ## Feature importance -  check hat m_gg and opening_angle are top
    importance = model.feature_importances_
    for f, imp in zip(features, importance):
        print(f"    {f}: {imp:.03f}")

    
    ## Accuracy metrics
    print("\n* Evaluating performance of events:")
    #result_events = all_df
    test_indices = X_test.index.tolist()
    result_events = pair_all_df.loc[test_indices].copy()

    # Count test set composition
    #test_signal = result_events['is_signal'].sum()
    #test_background = len(result_events) - test_signal
    #print(f"  Signal events: {test_signal} ({test_signal/len(result_events)*100:.1f}%)")
    #print(f"  Background events: {test_background} ({test_background/len(result_events)*100:.1f}%)")
    
    correct_predictions = 0
    total_pos_events = 0

    results = [] # Results list

    # Collect masses of identified pi0 candidates
    candidate_masses = []
    true_matches = []

    # Collect scores of identified pi0 candidates
    score_pos= []
    score_neg = []

    for _, evt in result_events.iterrows():
        # evt is a list object
        photons = [
            np.array([evt.E1, evt.px1, evt.py1, evt.pz1]),
            np.array([evt.E2, evt.px2, evt.py2, evt.pz2]),
            np.array([evt.E3, evt.px3, evt.py3, evt.pz3]),
        ]

        best_pair, score, mass = find_best_pi0_candidate(photons, model)
        
        # Create event_info dictionary for this event
        event_info ={
            'mass': mass,
            'score': score
        }

        # Save results
        results.append(event_info)

        # Couters
        if evt.is_signal:
            total_pos_events += 1
            if best_pair == evt.true_pi0_pair:
                correct_predictions += 1
                score_pos.append(score)
                status = "✓"
            else:
                score_neg.append(score)
                status = "✗"
        else:
            score_neg.append(score)
            status = "BG"

        truth = f"Postive pi0: {evt.true_pi0_pair}" if evt.is_signal else "Negative event"
        #print(f"Event {evt.event}: Best pair {best_pair}, score={score:.3f}, m={mass:.3f} | {status}")

        # mass collection
        if evt.is_signal and best_pair == evt.true_pi0_pair:
            candidate_masses.append(mass)
            true_matches.append(1)
        elif evt.is_signal:
            candidate_masses.append(mass)
            true_matches.append(0)

    if total_pos_events > 0:
        accuracy = correct_predictions / total_pos_events * 100 
        print(f"\nAccuracy on positive events: {accuracy:.1f}% ({correct_predictions}/{total_pos_events})")

    ## Check results list
    print(f"Collected {len(results)} events total")
    results_df = pd.DataFrame(results) # Covert to data frame
    print(results_df.head())

    ## Create score_pos and score_neg structure
    #print(rf'{len(score_pos)},     {len(score_neg)}')
    score_pos_df = pd.DataFrame({
        'score': score_pos,
        'type': 'signal_correct'
    })

    score_neg_df = pd.DataFrame({
        'score': score_neg,
        'type': 'background_or_wrong'
    })

    score_all_df = pd.concat([score_pos_df, score_neg_df], ignore_index=True) # Combining data from score_pos_df and score_neg_df
    #print(score_pos_df.head(5))
    #print(score_neg_df.head(5))
    #print(score_all_df.head(5))

    ## Plot mass vs. score
    m_pos = [m for m, match in zip(candidate_masses, true_matches) if match]
    m_neg = [m for m, match in zip(candidate_masses, true_matches) if not match]
    #print(m_pos)
    print(m_neg)
    var_list = [m_pos, m_neg]
    score_list = [score_pos, score_neg]
    var_str = [rf'$\pi^{0}$ Candidate Mass Distribution ({phys_ch})', 'Events', 'Invariant Mass (GeV)'] # [title, y_label, x_label]

    #plot_var_score(var_list, score_list, var_str, "Mass and Score", 'pi0_mass_score')

    ## ROC plot
    #plot_roc(score_list, rf'ROC Curve - $\pi^{0}$ Classifier ({phys_ch})', 'roc_curv')

    ## Check for overfitting
    plot_learning_curves(model, phys_ch, "learning_curves")
