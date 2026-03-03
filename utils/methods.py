import uproot
import awkward as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# =================================================================
# KLOE data, root file, ntuples
# =================================================================

def kloe_sample(input_str):

    """
    Two KLOE MC samples are used for labeling three photon finat state
    1. e+ e- -> phi -> eta gamma, eta -> pi+ pi- pi0 (2 gamma) + 1 gamma
    2. e+ e- -> pi+ pi- pi0 (2 gamma) + gamma (isr photon) 
    """
    print("="*50)
    print("Preparing KLOE data samples ...")
    print("="*50)


    #if phys_ch == "":
    #    print(f"Warning: No physical channel is assigned.")  
    f_nm = rf"./data/{input_str[0]}_sample.root"
    signal_root_file = uproot.open(f_nm) # Open signal root file
    input_br = input_str[1]
    print(rf"Opening {f_nm} ...")

    # Inspect keys
    for key, item in signal_root_file.items():
        print(f"{key}: {type(item).__name__}")

    signal_tree = signal_root_file[input_br] # Access the signal tree
    signal_df = signal_tree.arrays(library="pd") # Create data from the tree
    
    columns_signal_df = [col for col in signal_df.columns] # Original columns
    #print(f"columns_signal_df   {len(columns_signal_df)}, {columns_signal_df}")
    pho_nm_1 = ['Br_E1', 'Br_px1', 'Br_py1', 'Br_pz1']
    pho_nm_2 = ['Br_E2', 'Br_px2', 'Br_py2', 'Br_pz2']
    pho_nm_3 = ['Br_E3', 'Br_px3', 'Br_py3', 'Br_pz3']
    pho_nm_sum = pho_nm_1 + pho_nm_2 + pho_nm_3
    #print(var_nm)
    photon_var = [col for col in signal_df.columns if col in pho_nm_sum]
    #print(f"photon_var  {photon_var}")
    #photon_signal_df = signal_tree.arrays(photon_var, library="pd") 
    # Create (all, good, bad) signal data set with selected branches after the selection cuts: chi2 < 43
    photon_all_signal_df = signal_df[signal_df['Br_lagvalue_min_7C'] < 43][pho_nm_sum]
    photon_good_signal_df = signal_df[(signal_df['Br_lagvalue_min_7C'] < 43) & (signal_df['Br_recon_indx'] == 2) & (signal_df['Br_bkg_indx'])][pho_nm_sum]
    photon_bad_signal_df = signal_df[(signal_df['Br_lagvalue_min_7C'] < 43) & ~((signal_df['Br_recon_indx'] == 2) & (signal_df['Br_bkg_indx']))][pho_nm_sum]

    #print(photon_good_signal_df.head(5))

    # New column vectors
    pho_nm_1_new = ['E1', 'px1', 'py1', 'pz1']
    pho_nm_2_new = ['E2', 'px2', 'py2', 'pz2']
    pho_nm_3_new = ['E3', 'px3', 'py3', 'pz3']
    pho_nm_sum_new = pho_nm_1_new + pho_nm_2_new + pho_nm_3_new
    
    # Change column names
    if len(pho_nm_sum) == len(pho_nm_sum_new):
        # All signals
        #photon_all_signal_df.columns = pho_nm_sum_new
        #nb_all_signal = [i for i in range(len(photon_all_signal_df))] # Vector of number of all signal events
        ##photon_all_signal_df['event'] = nb_all_signal 
        #photon_all_signal_df.insert(0, 'event', nb_all_signal) # Add event column

        # Good signals
        photon_good_signal_df.columns = pho_nm_sum_new 
        nb_good_signal = [i for i in range(len(photon_good_signal_df))]  
        #photon_good_signal_df['event'] = nb_good_signal
        photon_good_signal_df.insert(0, 'event', nb_good_signal)  # Add event column

        is_signal_indx = [1] * len(nb_good_signal) 
        photon_good_signal_df['is_signal'] = is_signal_indx # Add is_signal_index

        true_pi0_pair_index = [(0, 1)] * len(nb_good_signal)
        photon_good_signal_df['true_pi0_pair'] = true_pi0_pair_index  # Add true_pi0_pair column
        #print(label_good_signal)
        

        # Bad signals
        photon_bad_signal_df.columns = pho_nm_sum_new
        nb_bad_signal = [i for i in range(len(photon_bad_signal_df))]
        #photon_bad_signal_df['event'] = nb_bad_signal
        photon_bad_signal_df.insert(0, 'event', nb_bad_signal) # Add event column 

        is_signal_indx = [0] * len(nb_bad_signal) 
        photon_bad_signal_df['is_signal'] = is_signal_indx # Add is_signal_index

        true_pi0_pair_index  = [(-1, -1)] * len(nb_bad_signal)
        photon_bad_signal_df['true_pi0_pair'] = true_pi0_pair_index # Add true_pi0_pair column

        # Combine good and bad photons
        photon_sum_signal_df = pd.concat([photon_good_signal_df, photon_bad_signal_df], ignore_index=True)
        # Add a column to identify source
        #photon_sum_signal_df['true_pi0_pair'] = [(0, 1)] * len(nb_good_signal) + [(-1, -1)] * len(nb_bad_signal)

        # Shuffle using numpy
        photon_sum_signal_df = photon_sum_signal_df.sample(frac=1).reset_index(drop=True)
    else:
        print(f"length mismatich!")
    
    #print(photon_good_signal_df.head(5))
    #print(f"Number of all signals = {len(photon_all_signal_df)}", nb_all_signal[:5])

    # Add an additional column for the signal encode: 1

    #print(signal_df.describe())
    #print(signal_df.head(6))

    return photon_sum_signal_df, photon_good_signal_df, photon_bad_signal_df

# =================================================================
# Testing data, simplified pi0 -> gamma gamma MC generator
# =================================================================
def MC_generation():
    """
    Create synthetic data for testing
    """

    print("="*50)
    print("SIMPLE XGBOOST FOR 3-PHOTON pi0 RECONSTRUCTION")
    print("="*50)

    np.random.seed(42)
    n_events = 1000
    print(f"Total number of events: {n_events}")

    #global synthetic_data
    synthetic_data = []
    for evt in range(n_events):
        # Signal: one pi0 + one extra photon
        if np.random.random() < 0.5:
            # pi0 at rest in its own frame, then boosted
            m_pi0 = 0.135
            #print(f"pi0 mass = {m_pi0} MeV")

            # Generate pi0 with some momentum
            pi0_pt = np.random.uniform(1, 10)
            pi0_eta = np.random.uniform(-1, 1)
            pi0_phi = np.random.uniform(-np.pi, np.pi)
            #print(f"pi0: (pt, eta, phi) = ({pi0_pt}, {pi0_eta}, {pi0_phi})")

            # Decay in rest frame: back-to-back photons
            # Then boost to lab frame
            # This is simplifed - in reality use proper decay generator
            # pi0 photon pair
            pi0_p = pi0_pt * np.cosh(pi0_eta) # momentum magnitude
            pi0_E = np.sqrt(m_pi0**2 + pi0_p**2)

            # Photon energies (simplified - equal sharing)
            e1_lab = pi0_E / 2
            e2_lab = pi0_E / 2

            # Approximate directions (small opening angle in lab)
            # Opening angle that guarantees m_gg = m_pi0
            dr = 2 * m_pi0 / pi0_E # approximation: theta = m/p

            # Photon directions
            phi1 = pi0_phi
            phi2 = pi0_phi + dr
            eta1 = pi0_eta
            eta2 = pi0_eta + dr / 2

            # Extra random photon
            e3 = np.random.uniform(0.5, 5)
            eta3 = np.random.uniform(-2, 2)
            phi3 = np.random.uniform(-np.pi, np.pi)

            # Convert (pt, eta, phi) to (E, px, py, pz)
            px1 = e1_lab * np.cos(phi1) / np.cosh(eta1)
            py1 = e1_lab * np.sin(phi1) / np.cosh(eta1)
            pz1 = e1_lab * np.sinh(eta1) / np.cosh(eta1)

            px2 = e2_lab * np.cos(phi2) / np.cosh(eta2)
            py2 = e2_lab * np.sin(phi2) / np.cosh(eta2)
            pz2 = e2_lab * np.sinh(eta2) / np.cosh(eta2)

            px3 = e3 * np.cos(phi3) / np.cosh(eta3)
            py3 = e3 * np.sin(phi3) / np.cosh(eta3)
            pz3 = e3 * np.sinh(eta3) / np.cosh(eta3)

            # Check m_gg
            ee = e1_lab + e2_lab
            pxx = px1 + px2
            pyy = py1 + py2
            pzz = pz1 + pz2
            mass_squared = ee**2 - (pxx**2 + pyy**2 + pzz**2)
            mass_pi0 = np.sqrt(np.maximum(0, mass_squared))
            #print(f"ee  {ee}; mass_pi0  {mass_pi0}")  

            # Data
            synthetic_data.append({
                'event': evt,
                'E1': e1_lab, 'px1': px1, 'py1': py1, 'pz1': pz1,
                'E2': e2_lab, 'px2': px2, 'py2': py2, 'pz2': pz2,
                'E3': e3, 'px3': px3, 'py3': py3, 'pz3': pz3,
                'is_signal': 1,
                'true_pi0_pair': (0, 1) # pion photon index
            })
            # print(f"e1 = {e1}, e1 = {e2}, dr = {dr}, e3 = {e3}, dr_extra = {dr_extra}")
        else:
            # Background: three random photon final state
            photons = []
            for k in range(3):
                e = np.random.uniform(0.5, 10)
                eta = np.random.uniform(-2, 2)
                phi = np.random.uniform(-np.pi, np.pi)
                photons.extend([e,
                                e * np.cos(phi) / np.cosh(eta),
                                e * np.sin(phi) / np.cosh(eta),
                                e * np.sinh(eta) / np.cosh(eta)   
                ])
        
            synthetic_data.append({
                'event': evt,
                'E1': photons[0], 'px1': photons[1], 'py1': photons[2], 'pz1': photons[3],
                'E2': photons[4], 'px2': photons[5], 'py2': photons[6], 'pz2': photons[7],
                'E3': photons[8], 'px3': photons[9], 'py3': photons[10], 'pz3': photons[11],
                'is_signal': 0,
                'true_pi0_pair': (-1, -1)
            })

    return synthetic_data


# =================================================================
# For a NEW 3-photon event, pick the best pi0 candidate
# =================================================================
def find_best_pi0_candidate(photon_4vectors, model):
    """
    photon_4vectors: list of 3 arrays, each [E, px, py, pz] or [E, pt, eta, phi]
    Returns: (best_pair_indices, probability, mass)
    """
    pairs = [(0,1), (0,2), (1,2)]
    candidates = []

    for i, j in pairs:
        # Calculate EXACT quantities from 4-vectors
        mass = inv_mass_4vector(photon_4vectors[i], photon_4vectors[j])
        #print(f"mass    {mass}; photon1_4vectors    {photon_4vectors[i]}; photon2_4vectors  {photon_4vectors[j]}")

        # True opening angle
        p1 = photon_4vectors[i]
        p2 = photon_4vectors[j]
        p1_mag = np.sqrt(np.maximum(0., p1[1]**2 + p1[2]**2 + p1[3]**2))
        p2_mag = np.sqrt(np.maximum(0., p2[1]**2 + p2[2]**2 + p2[3]**2))
        dot_product = p1[1] * p2[1] + p1[2] * p2[2] + p1[3] * p2[3]
        cos_theta = dot_product / (p1_mag * p2_mag + 1e-10) # 1e-10 avoid divide zero
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        #print(f"theta:  {theta}")

        # Energy asymmetry
        e1, e2 = p1[0], p2[0]
        asym = np.abs(e1 - e2) / (e1 + e2 + 1e-10)
        # print(f"e1: {e1}; e2: {e2}; asym: {asym}")

        # Predict
        proba = model.predict_proba([[mass, theta, cos_theta, asym]])[0, 1]

        candidates.append({
            'pair': (i, j),
            'score': proba,
            'mass': mass,
            'theta': theta
        })

        # Return the best candidate
        best = max(candidates, key=lambda x: x['score']) # best: The entire dictionary with the highest score
        # candidates: A list of dictionaries, each with a 'score' key
        # max(): Built-in Python function that finds the maximum value
        # key=lambda x: x['score']: Tells max() to use the 'score' value for comparison
        return best['pair'], best['score'], best['mass']

# =================================================================
# Bayesian Optimization
# =================================================================
def baye_opti(X_train, y_train):
    print("="*50)
    print("Searching for best model parameters ...")
    print("="*50)

    # Define search space
    search_spaces = {
        'n_estimators': Integer(100, 500),
        'max_depth': Integer(3, 10), 
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'min_child_weight': Integer(1, 10),
        'gamma': Real(0, 0.5),
        'reg_alpha': Real(0, 2),
        'reg_lambda': Real(0.5, 3)
    }

    # Bayesian optimization
    bayes_search = BayesSearchCV(
        estimator=xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            enable_categorical=True,
            random_state=42
        ),
        search_spaces=search_spaces,
        n_iter=50, # Number of optimization steps
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    bayes_search.fit(X_train, y_train)
    model_best = bayes_search.best_estimator_
    best_params = model_best.get_params()

    print("Best parameters:", bayes_search.best_params_)
    print("Best cross-validation AUC: {:.4f}".format(bayes_search.best_score_))
    
    #return bayes_search, model_best
    return best_params

# =================================================================
# Train XGBoost on EXACT 4-vector quantities
# =================================================================
def set_model_params(X_train, y_train):
    """
    Train classifier using EXACT invariant mass and true opening angle.
    This is MORE ACCURATE and SIMPLER than using Delta R approximations.
    """

    # Features: EXACT physics quantities from 4-vectors
    #features = ['m_gg', 'opening_angle', 'cos_theta', 'E_asym']
    #X = pair_df[features]
    #y = pair_df['is_pi0']


    ## Split
    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size = 0.2, random_state = 42
    #)

    # Ultra-simple XGBoost - shallow trees, fast training
    model = xgb.XGBClassifier(
        n_estimators = 100,
        max_depth = 3, # Shallow = fast, interpretable
        learning_rate = 0.1,
        objective = 'binary:logistic',
        eval_metric = 'auc', # parameter is used in machine learning models (like XGBoost, LightGBM, or sklearn-style APIs) to specify that you want to evaluate your model using AUC (Area Under the ROC Curve) metric.
        #use_label_encode=False,
        enable_categorical=True,
        random_state = 42
    )

    params = model.get_params()

    #return model, X_train, y_train, X_test, y_test
    return params

# =================================================================
# Physics METHOD
# =================================================================
def inv_mass_4vector(p1, p2):
    """
    Calculate diphoton invariant mass from two photon 4-vectors.

    Args:
        p1, p2: Arrays/lists of [E, px, py, pz] or [E, pt, eta, phi]

    Returns:
        Invariant mass in GeV
    """

    if len(p1) == 4 and len(p2) == 4:
        e = p1[0] + p2[0]
        px = p1[1] + p2[1]
        py = p1[2] + p2[2]
        pz = p1[3] + p2[3]
        mass_squared = e**2 - (px**2 + py**2 + pz**2)
        # Ensure non-negative before sqrt
        return np.sqrt(np.maximum(0., mass_squared))
        #if (mass_squared < 0):
            #print(f"mass_squared    {mass_squared}")
        #return np.sqrt(mass_squared)
    else:   
        # Use your experiment's Lorentz vector class
        # (e.g., ROOT.TLorentzVector, vector.obj, etc.)
        return (p1 + p2).M()
    
def prepare_3photon_paris(df_events):
    """
    Convert 3-photon events into training paris with EXACT invariant masses.

    Assumes your DataFrame has columns:
    E1, px1, py1, pz1,  # OR E1, pt1, eta1, phi1
    E2, px2, py2, pz2,
    E3, px3, py3, pz3,
    is_signal, true_pi0_pair
    """

    print("="*50)
    print("Creating photon pairs with EXACT quantities: invariant masses ...")
    print("="*50)

    pairs = []

    for _, evt in df_events.iterrows():
        # Get 4-vector for all 3 photons
        # ADAPT THIS TO YOUR EXACT COLUMN NAMES
        photons = [
            np.array([evt.E1, evt.px1, evt.py1, evt.pz1]), # [E, px, py, pz]
            np.array([evt.E2, evt.px2, evt.py2, evt.pz2]),
            np.array([evt.E3, evt.px3, evt.py3, evt.pz3])
        ]

        # All 3 possible pairs
        pair_indices = [(0,1), (0,2), (1,2)]

        #print(f"{pair_indices},{type(pair_indices)}, {type(df_events)}")
        #print(photon)

        for i, j in pair_indices:
            # Calculate EXACT invariant mass from 4-vectors
            mass = inv_mass_4vector(photons[i], photons[j])
            #print(f"mass = {mass}")

            # Opening angle
            p1_mag = np.sqrt(np.maximum(0., photons[i][1]**2 + photons[i][2]**2 + photons[i][3]**2))
            p2_mag = np.sqrt(np.maximum(0., photons[j][1]**2 + photons[j][2]**2 + photons[j][3]**2))
            dot_product = photons[i][1] * photons[j][1] + photons[i][2] * photons[j][2] + photons[i][3] * photons[j][3]
            cos_theta = dot_product / (p1_mag * p2_mag + 1e-10) # 1e-10 avoid divide zero
            theta = np.arccos(np.clip(cos_theta, -1, 1))
            #print(f"p1_mag = {p1_mag}, p2_mag = {p2_mag}")
            #print(f"dot_product = {dot_product}, cos_theta = {cos_theta}")
            #print(f"{np.clip(cos_theta, -1, -1)}")
            #print(f"theta = {theta}")

            # Energy asymmetry
            e1 = photons[i][0]
            e2 = photons[j][0]
            e_asym = np.abs(e1 - e2) / (e1 + e2 + 1e-10)
            #print(f"p_asym = {e_asym}")

            # Energy ratio
            e_ratio = min(e1, e2) / max(e1, e2)

            # Energy diff.
            e_diff = np.abs(e1 - e2)
            
            # Minimum energy angle
            e_min_x_angle = min(e1, e2) * theta

            # Energy asymmetry angle
            asym_x_angle = e_asym * theta

            # Is this the correct pi0 pair? (require truth info)
            is_pi0 = 0
            if 'is_signal' in evt and evt.is_signal == 1:
                if 'true_pi0_pair' in evt:
                    is_pi0 = 1 if (i, j) == evt.true_pi0_pair else 0
                else:
                    # If you don't have exact pair truth,
                    # assume the pair with mass closest to 0.135 GeV is correct
                    is_pi0 = 1 if abs(mass - 0.135) < 0.015 else 0

            # Data insertion
            pairs.append({
                'event': evt.event,
                'pair_id': f"{evt.event}_{i}{j}",
                'm_gg': mass,
                'opening_angle': theta, # opening angle in radians
                'cos_theta': cos_theta,
                'E_asym': e_asym,
                'e_min_x_angle': e_min_x_angle,
                'E1': e1,
                'E2': e2,
                #'asym_x_angle': asym_x_angle,
                #'E_diff': e_diff,
                'E_ratio': e_ratio,
                'is_pi0': is_pi0
            })
    
    return pd.DataFrame(pairs)

# =================================================================
# PLOT METHOD
# =================================================================

# Plot (all, good, bad) comparison
def plot_compr_hist(df_set, rows, bins, plot_title, plot_nm):

    all_df = df_set[0]
    good_df = df_set[1]
    bad_df = df_set[2]

    ##  S/B ratio
    S = len(good_df)
    B = len(bad_df)
    S_purity = S / (S + B)
    print(f"Total events: {len(all_df)}, good events: {len(good_df)}, bad events: {len(bad_df)}")
    print(f"S/sqrt(S+B): {S / np.sqrt(S + B):.2f}")
    print(f"S_purity: {S_purity:.2f}")

    ## Check col_len
    col_len = len(all_df.columns) # length of columns of df

    if (col_len < 0):
        # negative
        print(f"Negative col_len ({col_len})")
        return
    elif (col_len == 0):
        # zero col_len
        print(f"Zero col_len ({col_len})")
    else:
        # postive
        if (col_len % 2 == 0):
            # even case
            print(f"good events col_len ({col_len})")
        else:
            # odd or not integer
            print(all_df.columns)
            print(f"Odd column length or none integer column length ({col_len}). Not plot is created!")
            return

    # Create subplot grid
    plot_col = int(col_len / rows) # number of rows and columns to the plot
    fig, axes = plt.subplots(rows, plot_col, figsize=(16, 10)) # rows and columns to subplots
    fig.suptitle(plot_title, fontsize=16, y=1.02)

    # Flatten axes array for easy iteration
    axes = axes.flatten()
    columns = all_df.columns
    
    #for i, label in enumerate(columns_df[:col_len]):
    for i, label in enumerate(columns):
        print(i, label)
        # desity=True normalized
        positive_good_df = good_df[good_df[label] > 0.2][label]
        positive_bad_df = bad_df[bad_df[label] > 0.2][label]
        positive_all_df = all_df[all_df[label] > 0.2][label]

        if label in ['E1', 'E2', 'E3']:
            unit = fr'[$\mathrm{{MeV}}$]'
        elif label in ['m_gg']:
            unit = fr'[$\mathrm{{MeV}}/\mathrm{{c}}^{2}$]'
        elif label in ['px1', 'py1', 'pz1', 'px2', 'py2', 'pz2', 'px3', 'py3', 'pz3', 'E_asym']:
            unit = fr'[$\mathrm{{MeV}}/\mathrm{{c}}$]'
        else:
            unit = ""
            #rint("AU")

        n1, bin_edges1, patches1 = axes[i].hist([positive_good_df, positive_bad_df], 
                     color=['green', 'blue'], 
                     bins=bins, 
                     label=[f'Good', f'Bad'], 
                     density=False, 
                     edgecolor=['green', 'blue'],
                     linewidth=1, 
                     alpha=0.5,
                     histtype='stepfilled' # Filled histograms
                     )

        n2, bin_edges2, patches2 = axes[i].hist(positive_all_df, 
                     color=['red'], 
                     bins=bin_edges1, 
                     label='All', 
                     density=False, 
                     edgecolor='red',
                     linewidth=1, 
                     alpha=0.5,
                     histtype='step' # Filled histograms
                     )
        
        

        bin_width = bin_edges2[1] - bin_edges2[0]
        #print(f"bin_width: {bin_width:.2f}")
 
        axes[i].set_xlabel(label + ' ' + unit)
        #axes[i].set_ylabel(fr'Events / {bin_width:.1f} {unit}', fontsize=14)
        axes[i].set_ylabel(fr'Events', fontsize=14)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='best', fontsize=14) 
    
    #plt.title(plot_title)
    plt.tight_layout()
    plt.savefig('./plots/' + plot_nm + '_compr.png', dpi=300, bbox_inches='tight')
    #pt.show(block=False)
    plt.show()
    plt.close()


# Plot single variable
def plot_var(array, var_nm, phys_ch):
    print(f"Plotting ... {var_nm}")
    fig = figsize=(16, 10)
    plt.hist(array, 
             color='green', 
             bins=100, 
             density=False, 
             edgecolor='black', 
             alpha=0.7, 
             label=r'Reconstructed $\pi^{0}$',
             histtype='stepfilled'
             )
    plt.xlabel(r'$m_{\gamma\gamma}$ [GeV/c$^2$]')                                  
    plt.ylabel('Events')
    plt.title(fr'Mass Distribution of true $\pi^{0}$ (n={len(array)}) {phys_ch}')
    # combine into one legend
    #plt.legend(loc='best', fontsize=8, frameon=True, fancybox=True, shadow=True,
    #           title=f'π⁰ Mass Distribution (n={len(array)})\nTrue π⁰ events'
    #)
    #plt.legend(loc='best', fontsize=8, title=f'π⁰ Mass Distribution (n={len(array)})') 
    plt.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.savefig('./plots/signal_pi0.png')
    #plt.show(block=False)
    #plt.show()
    #plt.close()

# Plot individual feature set
def plot_hist(df, rows, bins, plot_nm):
    
    col_len = len(df.columns) # length of columns of df

    # check col_len
    if (col_len < 0):
        # negative
        print(f"Negative col_len ({col_len})")
        return
    elif (col_len == 0):
        # zero col_len
        print(f"Zero col_len ({col_len})")
    else:
        # postive
        if (col_len % 2 == 0):
            # even case
            print(f"good events col_len ({col_len})")
        else:
            # odd or not integer
            print(f"Odd column length or none integer column length ({col_len})! No plot is created!")
            return

    # Create subplot grid
    plot_col = int(col_len / rows) # number of rows and columns to the plot
    fig, axes = plt.subplots(rows, plot_col, figsize=(16, 10)) # rows and columns to subplots
    fig.suptitle(plot_nm, fontsize=16, y=1.02)
    colors = plt.cm.tab10(np.linspace(0, 1, col_len))  # Generate distinct colors
    #color = ["blue", "black", "red", "yellow", "purple", "green", "orange", "brown", "gray", "cyan"]
    #print(f"columns list ({col_len}); plot {rows}x{plot_col}")

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    #for i, label in enumerate(columns_df[:col_len]):
    for i, label in enumerate(df.columns):
        #print(i, label)
        # desity=True normalized
        positive_data = df[df[label] > 0.2][label]
        axes[i].hist(positive_data, 
                     color=colors[i], 
                     bins=bins, 
                     label=label, 
                     density=False, 
                     edgecolor=colors[i],
                     linewidth=1, 
                     alpha=0.7,
                     histtype='stepfilled' # Filled histograms
                     )
        #axes[i].hist(df[label], color=colors[i], bins=bins, label=label, density=False, edgecolor='black', alpha=0.7)
        #axes[i].set_title(label, fontsize=12)
        if label in ['E1', 'E2', 'E3']:
            unit = fr'$\mathrm{{MeV}}/\mathrm{{c}}^{2}$'
        elif label in ['px1', 'py1', 'pz1', 'px2', 'py2', 'pz2', 'px3', 'py3', 'pz3']:   
            unit = fr'$\mathrm{{MeV}}/\mathrm{{c}}$'
        else:
            print("AU")
 
        axes[i].set_xlabel(label + ' [' + unit + ']')
        axes[i].set_ylabel(None)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='best', fontsize=14) 
        #axes[i].set_yscale('log')  # <-- LOG SCALE ON Y-AXIS
        
        #if (i == 0):
        #    break
        
    plt.title(plot_nm)
    plt.tight_layout()
    plt.savefig('./plots/' + plot_nm + '.png', dpi=300, bbox_inches='tight')
    #plt.show(block=False)
    plt.show()
    #plt.close()

# Plot feature-feature
def plot_feature_pairs(df, plot_title, plot_nm):
    print('Plotting feature pairs')
    df_tmp = df.drop(['event', 'cos_theta', 'pair_id'], axis=1)
    feature_columns = df_tmp.columns
    print(feature_columns)
    print(df.describe())


    g = sns.pairplot(df_tmp[feature_columns], # Data
                     hue = 'is_pi0', # Color grouping, points by the values in the 'is_pi0' column
                     palette={1: 'blue', 0: 'red'}, # 3. colors     
                     diag_kind='hist', # Diagonal plot type
                     plot_kws={'alpha': 0.5, 's': 10}, # Scatter plot options
                     diag_kws={'alpha': 0.7, 'edgecolor': 'black'} # Histogram options  
    )
    g.figure.suptitle(plot_title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig('./plots/' + plot_nm, dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()

# Plot feature-target     
def plot_feature_target(target_corr, plot_title, plot_nm):
    print('here plotting ...')
    plt.figure(figsize=(10, 6))
    colors = ['green' if c > 0  else 'red' for c in target_corr.values]
    plt.bar(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xticks(range(len(target_corr)), target_corr.index, rotation=45, ha='right')
    plt.ylabel(rf'Correlation with true $\pi^{0}$')
    plt.title(plot_title)
    #plt.title(rf'Feature Importance: Correlation with true $\pi^{0}$')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (feat, corr) in enumerate(target_corr.items()):
        #print (i, feat, corr)
        plt.text(i, corr + (0.02 if corr > 0 else -0.05),
                    f'{corr:.2f}', ha='center', va='bottom' if corr > 0 else 'top')
        
    plt.tight_layout()
    plt.savefig('./plots/' + plot_nm + '.png', dpi=300, bbox_inches='tight')
    #plt.show(block=False)
    plt.close()

# Plot variable vs. score
def plot_var_score(var_list, score_list, var_str, plot_title, plot_nm):
    print("Plotting variable vs. score ...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle(plot_title, fontsize=16, y=1.02)

    titles = [var_str[0], 'Score']
    y_labels = [var_str[1], 'Events']
    x_labels = [var_str[2], 'Score']

    for i in range(2):
        if (i == 0): # mass distributions
            n, bin_edges, patches = axes[i].hist(var_list,
                                                bins=200, 
                                                alpha=0.5, 
                                                label=['Correctly identified', 'Wrongly identified'],
                                                color=['green', 'black'],
                                                density=False,
                                                linewidth=1,
                                                histtype='stepfilled'
                                                )
            
            axes[i].set_title(fr'{titles[i]}', fontsize=18)
            #axes[i].set_xlim(50, 200) # Set x-axis range in [MeV/c^2]
            axes[i].set_xlabel(fr'{x_labels[i]}', fontsize=14)
            axes[i].set_ylabel(fr'{y_labels[i]}')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='best', fontsize=14)
            #axes[i].set_yscale('log')
            axes[i].axvline(x=135, color='black', linestyle='--', label='True pi0 mass')
        else:
            n, bin_edges, patches = axes[i].hist(score_list,
                                                bins=100, 
                                                alpha=0.5, 
                                                label=['Signal', 'Background'],
                                                color=['blue', 'red'],
                                                density=False,
                                                linewidth=1,
                                                histtype='step'
                                                )
            axes[i].set_title(fr'{titles[i]}', fontsize=18)
            #axes[i].set_xlim(0, 0.2) # Set x-axis range from 0 to 0.2
            axes[i].set_xlabel(fr'{x_labels[i]}', fontsize=14)
            axes[i].set_ylabel(fr'{y_labels[i]}')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='best', fontsize=14)
            axes[i].axvline(x=0.5, color='black', linestyle='--', label='True pi0 mass')
            axes[i].set_yscale('log')

    plt.tight_layout()
    plt.savefig(rf'./plots/{plot_nm}.png')
    plt.show()

## Plot ROC curv (Performance)
def plot_roc(score_list, plot_title, plot_nm):
    print("Plotting ROC curv ...")
    """
    - score_pos: scores from positive class (correctly identified pi0)
    - score_neg: scores from negative class (wrongly identified pi0)
    """
    score_pos = score_list[0]
    score_neg = score_list[1]
    y_true = [1] * len(score_pos) + [0] * len(score_neg)
    y_score = score_pos + score_neg
    #print(len(y_true), y_true)
    #print(y_score)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr,
             tpr,
             color='black',
             lw=2,
             label=f'ROC AUC: {roc_auc:.4f}'
             )
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(plot_title)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add statistics
    textstr = f'Positive: {len(score_pos)} events\nBackground: {len(score_neg)} events'
    plt.text(0.6, 0.2, textstr, fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5",
             facecolor='yellow',
             alpha=0.3))
    plt.tight_layout()
    plt.savefig(f'./plots/{plot_nm}.png', dpi=300)
    plt.show()

# Plot learning curves (Check for overfitting)
def plot_learning_curves(model, phys_ch, plot_nm):
    """
    Plot training vs validation performance over boosting rounds
    This shows if model is overfitting
    """
    print("Plotting learning curves...")

    results = model.evals_result()

    train_auc = results['validation_0']['auc']
    val_auc = results['validation_1']['auc']
    #print(train_auc)

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))

    # AUC over rounds
    axes.plot(train_auc, 'b-', label='Training', linewidth=2)
    axes.plot(val_auc, 'r-', label='Validation', linewidth=2)
    axes.set_xlabel('Boosting Round', fontsize=14)
    axes.set_ylabel('AUC', fontsize=14)
    axes.set_title(rf'Learning Curves - AUC', fontsize=14)
    axes.legend()
    axes.grid(True, alpha=0.3)

    plt.suptitle(f'Model Validation - {phys_ch}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'./plots/{plot_nm}.png', dpi=300)
    plt.show()

    # Print diagnostics
    final_gap = train_auc[-1] - val_auc[-1]
    print(f"\n📊 Validation Diagnostics:")
    print(f"  Final Training AUC: {train_auc[-1]:.4f}")
    print(f"  Final Validation AUC: {val_auc[-1]:.4f}")
    print(f"  Gap: {final_gap:.4f}")
    
    if final_gap > 0.05:
        print("  ⚠️  WARNING: Possible overfitting!")
    elif final_gap > 0.02:
        print("  ⚠️  Caution: Moderate gap")
    else:
        print("  ✅ Good generalization!")