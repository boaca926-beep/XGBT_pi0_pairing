import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import joblib
import uproot
import numpy as np
import pandas as pd
from plots import plot_compr_hist, plot_var, plot_feature_pairs, plot_feature_target
from training.config import prepare_3photon_paris, prepare_kine_var
from sklearn.model_selection import train_test_split
import random

# Create and split dataset
# Signal: 
#   1. etagam 
#   2. isr3pi
# Background:
#   2.


def load_bkg_dataset():
    print("Loading background dataset ...")

##   
def load_signal_dataset():
    print("Loading signal dataset ...")

    # True positive and true negative events after selection
    br_nm = ['Br_E1', 'Br_px1', 'Br_py1', 'Br_pz1', 
             'Br_E2', 'Br_px2', 'Br_py2', 'Br_pz2', 
             'Br_E3', 'Br_px3', 'Br_py3', 'Br_pz3',
             'Br_m3pi', 'Br_lagvalue_min_7C']

    # Selection cut
    pos_df = df[(df['Br_lagvalue_min_7C'] < 43) & (df['Br_recon_indx'] == 2) & (df['Br_bkg_indx'])][br_nm]
    neg_df = df[(df['Br_lagvalue_min_7C'] < 43) & ~((df['Br_recon_indx'] == 2) & (df['Br_bkg_indx']))][br_nm]
    #print(pos_df.columns)
    
    para_nm = ['E1', 'px1', 'py1', 'pz1', 
               'E2', 'px2', 'py2', 'pz2', 
               'E3', 'px3', 'py3', 'pz3', 
               'm3pi', 'lagvalue_min_7C']
    
    # Combine postive and negative dataset to all_df
    if len(br_nm) == len(para_nm):
        
        # Change column names
        pos_df.columns = para_nm
        neg_df.columns = para_nm
    
        # True positive
        nb_pos = [i for i in range(len(pos_df))]  
        pos_df.insert(0, 'event', nb_pos)  # Add event column

        is_signal_indx = [1] * len(nb_pos) 
        pos_df['is_signal'] = is_signal_indx # Add is_signal column

        true_pi0_index = [(0, 1)] * len(nb_pos)
        pos_df['true_pi0_pair'] = true_pi0_index  # Add true_pi0_pair column
        
        # True negative
        nb_neg = [i for i in range(len(neg_df))]
        neg_df.insert(0, 'event', nb_neg) # Add event column 

        is_signal_indx = [0] * len(nb_neg) 
        neg_df['is_signal'] = is_signal_indx # Add is_signal column

        true_pi0_index  = [(-1, -1)] * len(nb_neg)
        neg_df['true_pi0_pair'] = true_pi0_index # Add true_pi0_pair column

        # pos + neg dataset
        all_df = pd.concat([pos_df, neg_df], ignore_index=True)

        # Shuffle using numpy
        all_df = all_df.sample(frac=1).reset_index(drop=True)

    else:
        raise ValueError("Array length mismatch - cannot proceed")

    return all_df, pos_df, neg_df

##
def data_splitting():
    print('Splitting dataset ...')

    # First split: separate test set 20%
    all_df_trainval, all_df_test = train_test_split(
        all_df, test_size=0.2, random_state=42
    )

    # Second split: separte validation from training (20% of total)
    all_df_train, all_df_val = train_test_split(
        all_df_trainval, test_size=0.25, random_state=42
    )

    print(f"Training events:   {len(all_df_train)} ({len(all_df_train)/len(all_df)*100:.1f}%)")
    print(f"Validation events: {len(all_df_val)} ({len(all_df_val)/len(all_df)*100:.1f}%)")
    print(f"Test events:       {len(all_df_test)} ({len(all_df_test)/len(all_df)*100:.1f}%)")

    print("CREATING PAIRS FROM EACH SPLIT")

    pair_train = prepare_3photon_paris(all_df_train)
    pair_val = prepare_3photon_paris(all_df_val)
    pair_test = prepare_3photon_paris(all_df_test)

    print(f"Training pairs:   {len(pair_train)}")
    print(f"Validation pairs: {len(pair_val)}")
    print(f"Test pairs:       {len(pair_test)}")

    ## Step 3: Prepare features for training
    features = ['m_gg', 'opening_angle', 'cos_theta', 'E_asym', 'e_min_x_angle', 'E1', 'E2', 'E3', 'asym_x_angle', 'E_diff']
    X_train = pair_train[features]
    y_train = pair_train['is_pi0']
    X_val = pair_val[features]
    y_val = pair_val['is_pi0']
    X_test = pair_test[features]
    y_test = pair_test['is_pi0']

    print(f"\n✅ Data ready for training:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape:   {X_val.shape}")
    print(f"  X_test shape:  {X_test.shape}")

    return all_df_train, all_df_val, all_df_test, X_train, y_train, X_val, y_val, X_test, y_test

##
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
 
    # Load X_test, y_test
    X_test = joblib.load(os.path.join(input_data_dir, f'X_test_{data_type}.pkl'))
    y_test = joblib.load(os.path.join(input_data_dir, f'y_test_{data_type}.pkl'))

    # Load all_df, pos_df, neg_df
    all_df = joblib.load(os.path.join(input_data_dir, f'all_df_{data_type}.pkl'))
    pos_df = joblib.load(os.path.join(input_data_dir, f'pos_df_{data_type}.pkl'))
    neg_df = joblib.load(os.path.join(input_data_dir, f'neg_df_{data_type}.pkl'))

    # Load pi0_all_df, pi0_pos_df, pi0_neg_df
    pi0_all_df = joblib.load(os.path.join(input_data_dir, f'pi0_all_df_{data_type}.pkl'))
    pi0_pos_df = joblib.load(os.path.join(input_data_dir, f'pi0_pos_df_{data_type}.pkl'))
    pi0_neg_df = joblib.load(os.path.join(input_data_dir, f'pi0_neg_df_{data_type}.pkl'))

 
    return X_train, y_train, X_val, y_val, X_test, y_test, all_df, pos_df, neg_df, pi0_all_df, pi0_pos_df, pi0_neg_df

if __name__ == '__main__':

    # Phys map
    phys_map = {
        # Signal channels
        #'etagam':{
        #    'br_nm': 'TETAGAM',
        #    'br_title': rf"$e^{{+}}e^{{-}}\to\phi\to\eta\gamma$",
        #    'category': 'signal'
        #},
        #'isr3pi':{
        #    'br_nm': 'TISR3PI_SIG',
        #    'br_title': rf"$e^{{+}}e^{{-}}\to\pi^{{+}}\pi^{{-}}\pi^{{0}}\gamma$",
        #    'category': 'signal'
        #},

        # Background channels
        'ksl':{
            'br_nm': 'TKSL',
            'br_title': rf"$e^{{+}}e^{{-}}\to\phi\to K_{{S}}K_{{L}}$",
            'category': 'background'
        }
    }

    phys_map_combined = {
        # Combined channels
        'combined':{
            'br_nm': 'TCOMB',
            'br_title': rf"$\pi^{{+}}\pi^{{-}}\pi^{{0}}\gamma+\eta\gamma$",
            'category': 'combined'
        }
    }

    # Loop over dataset
    for data_type, info in phys_map.items():
        #============================================================
        # PREPARATION (input & output folder, input data type)
        #============================================================
        br_nm = info['br_nm']
        br_title = info['br_title']
        print(br_nm, br_title)

        # Create output folder
        data_dir = rf'./dataset' # store all_df, (X, y)_{train, val, test}
        plot_dir = rf'./plots'
        os.makedirs(data_dir , exist_ok=True)
        os.makedirs(plot_dir , exist_ok=True)

        #============================================================
        # LOAD INPUT ROOT FILES
        #============================================================
        f_nm = rf"../data/{data_type}_sample.root"
        signal_root_file = uproot.open(f_nm)
        #try:
        #    file = uproot.open()
        #    print('All keys:', file.keys())
        #except Execption as e:
        #    print('Error: ', e)

        print(rf"Opening {f_nm} ...")

        # Create data frame
        for key, item in signal_root_file.items(): # Inspect keys
            print(f"{key}: {type(item).__name__}")
        
        tree = signal_root_file[br_nm] # Access the signal tree
        df = tree.arrays(library="pd") 
        #print(df.describe())

        #============================================================
        # CREAT PHOTON 4-MOM AND PAIRED DATASET FOR TRAINING, VAL AND TEST
        #============================================================
        if info['category'] == 'signal':
            all_df, pos_df, neg_df = load_signal_dataset()
            print(f"all_df columns: {all_df.columns}")
        elif info['category'] == 'background':
            all_df = load_bkg_dataset()
            print("loading dataset for background ...")

        #print(f'len bad     ', (all_df['true_pi0_pair'] == (-1, -1)).sum())
        print(all_df['true_pi0_pair'].value_counts(), f'len good    {len(pos_df)}')

        # kinematic variables
        kine_all_df = all_df[['m3pi', 'lagvalue_min_7C', 'is_signal']]
        kine_pos_df = kine_all_df[kine_all_df['is_signal'] == 1]
        kine_neg_df = kine_all_df[~(kine_all_df['is_signal'] == 1)]
        #print(kine_all_df.head(5))
        #print(kine_pos_df)
        #print(kine_neg_df)

        #print(kine_all_df.columns)
        #print(kine_all_df.head(5))
        #kine_pos_df = kine_all_df[kine_all_df['is_pi0'] == 1]
        #kine_neg_df = kine_all_df[kine_all_df['is_pi0'] == 0]
        #print(f"total events    {len(kine_all_df)}\ngood events     {len(kine_pos_df)}\nbad events     {len(kine_neg_df)}")
        #print(f"good ratio  {len(kine_pos_df)/len(kine_all_df):.2f}")

        # pi0 candidates
        pi0_all_df= prepare_3photon_paris(all_df)
        pi0_pos_df = pi0_all_df[pi0_all_df['is_pi0'] == 1]
        pi0_neg_df = pi0_all_df[pi0_all_df['is_pi0'] == 0]
        print(f"total pi0 candidates    {len(pi0_all_df)}\ngood candidates     {len(pi0_pos_df)}\nbad candidates     {len(pi0_neg_df)}")
        print(f"good ratio  {len(pi0_pos_df)/len(pi0_all_df):.2f}")
        #print(f"pi0_all_df columns: {pi0_all_df.columns}")

        # Data spliting
        all_df_train, all_df_val, all_df_test, X_train, y_train, X_val, y_val, X_test, y_test = data_splitting()
        
        #============================================================
        # SAVE DATA
        #============================================================
        # Save phys. channel map
        joblib.dump(phys_map, f'{data_dir}/phys_map_indiv.pkl')
        joblib.dump(phys_map_combined, f'{data_dir}/phys_map_combined.pkl')

        # Save postive, negative and sum dataset
        # gamma 4-momentum
        joblib.dump(all_df, f'{data_dir}/all_df_{data_type}.pkl')
        joblib.dump(pos_df, f'{data_dir}/pos_df_{data_type}.pkl')
        joblib.dump(neg_df, f'{data_dir}/neg_df_{data_type}.pkl')

        # p0 features
        joblib.dump(pi0_all_df, f'{data_dir}/pi0_all_df_{data_type}.pkl')
        joblib.dump(pi0_pos_df, f'{data_dir}/pi0_pos_df_{data_type}.pkl')
        joblib.dump(pi0_neg_df, f'{data_dir}/pi0_neg_df_{data_type}.pkl')

        # kine. var
        joblib.dump(kine_all_df, f'{data_dir}/kine_all_df_{data_type}.pkl')
        joblib.dump(kine_pos_df, f'{data_dir}/kine_pos_df_{data_type}.pkl')
        joblib.dump(kine_neg_df, f'{data_dir}/kine_neg_df_{data_type}.pkl')

        # Save splitted dataset
        # train
        joblib.dump(all_df_train, f'{data_dir}/all_df_train_{data_type}.pkl')
        joblib.dump(X_train, f'{data_dir}/X_train_{data_type}.pkl')
        joblib.dump(y_train, f'{data_dir}/y_train_{data_type}.pkl')

        # validation
        joblib.dump(all_df_val, f'{data_dir}/all_df_val_{data_type}.pkl')
        joblib.dump(X_val, f'{data_dir}/X_val_{data_type}.pkl')
        joblib.dump(y_val, f'{data_dir}/y_val_{data_type}.pkl')

        # test
        joblib.dump(all_df_test, f'{data_dir}/all_df_test_{data_type}.pkl')
        joblib.dump(X_test, f'{data_dir}/X_test_{data_type}.pkl')
        joblib.dump(y_test, f'{data_dir}/y_test_{data_type}.pkl')

        
        # Save feature names for reference
        #print(f"features: {X_val.columns}")

        #with open(f'{data_dir}/feature_name.txt', 'w') as f:
        #          for feat in X_val.columns:
        #            f.write(f"{feat}\n")

    r'''
    # Combine dataset
    print("Combining dataset ...")

    input_data_dir = './dataset'

    X_train_list = []
    y_train_list = []

    X_val_list = []
    y_val_list = []

    X_test_list = []
    y_test_list = []
    
    all_df_list = []
    pos_df_list = []
    neg_df_list = []

    pi0_all_df_list = []
    pi0_pos_df_list = []
    pi0_neg_df_list = []

    kine_all_df_list = []
    kine_pos_df_list = []
    kine_neg_df_list = []

    for data_type, info in phys_map.items():
        br_nm = info['br_nm']
        br_title = info['br_title']
        category = info['category']
        #print(f"Inspecting dataset {data_type}; {br_nm}; {br_title}; {category}")  
        print(info)

        X_train, y_train, X_val, y_val, X_test, y_test, all_df, pos_df, neg_df, pi0_all_df, pi0_pos_df, pi0_neg_df = load_dataset(data_type)

        X_train_list.append(X_train)
        y_train_list.append(y_train)

        X_val_list.append(X_val)
        y_val_list.append(y_val)

        X_test_list.append(X_test)
        y_test_list.append(y_test)

        all_df_list.append(all_df)
        pos_df_list.append(pos_df)
        neg_df_list.append(neg_df)

        pi0_all_df_list.append(pi0_all_df)
        pi0_pos_df_list.append(pi0_pos_df)
        pi0_neg_df_list.append(pi0_neg_df)

        kine_all_df_list.append(kine_all_df)
        kine_pos_df_list.append(kine_pos_df)
        kine_neg_df_list.append(kine_neg_df)

    X_train_combined = pd.concat(X_train_list, ignore_index=True)
    y_train_combined = pd.concat(y_train_list, ignore_index=True)
    
    X_val_combined = pd.concat(X_val_list, ignore_index=True)
    y_val_combined = pd.concat(y_val_list, ignore_index=True)

    X_test_combined = pd.concat(X_test_list, ignore_index=True)
    y_test_combined = pd.concat(y_test_list, ignore_index=True)

    all_df_combined = pd.concat(all_df_list, ignore_index=True)
    pos_df_combined = pd.concat(pos_df_list, ignore_index=True)
    neg_df_combined = pd.concat(neg_df_list, ignore_index=True)

    pi0_all_df_combined = pd.concat(pi0_all_df_list, ignore_index=True)
    pi0_pos_df_combined = pd.concat(pi0_pos_df_list, ignore_index=True)
    pi0_neg_df_combined = pd.concat(pi0_neg_df_list, ignore_index=True)

    kine_all_df_combined = pd.concat(kine_all_df_list, ignore_index=True)
    kine_pos_df_combined = pd.concat(kine_pos_df_list, ignore_index=True)
    kine_neg_df_combined = pd.concat(kine_neg_df_list, ignore_index=True)

    # Shuffle together using the same index
    shuffled_idx = np.random.permutation(len(X_train_combined))
    X_train_combined = X_train_combined.iloc[shuffled_idx].reset_index(drop=True)
    y_train_combined = y_train_combined.iloc[shuffled_idx].reset_index(drop=True)

    shuffled_idx = np.random.permutation(len(X_val_combined))
    X_val_combined = X_val_combined.iloc[shuffled_idx].reset_index(drop=True)
    y_val_combined = y_val_combined.iloc[shuffled_idx].reset_index(drop=True)

    shuffled_idx = np.random.permutation(len(X_test_combined))
    X_test_combined = X_test_combined.iloc[shuffled_idx].reset_index(drop=True)
    y_test_combined = y_test_combined.iloc[shuffled_idx].reset_index(drop=True)

    #
    shuffled_idx = np.random.permutation(len(all_df_combined))
    all_df_combined = all_df_combined.iloc[shuffled_idx].reset_index(drop=True)

    shuffled_idx = np.random.permutation(len(pos_df_combined))
    pos_df_combined = pos_df_combined.iloc[shuffled_idx].reset_index(drop=True)

    shuffled_idx = np.random.permutation(len(neg_df_combined))
    neg_df_combined = neg_df_combined.iloc[shuffled_idx].reset_index(drop=True)

    #
    shuffled_idx = np.random.permutation(len(pi0_all_df_combined))
    pi0_all_df_combined = pi0_all_df_combined.iloc[shuffled_idx].reset_index(drop=True)

    shuffled_idx = np.random.permutation(len(pi0_pos_df_combined))
    pi0_pos_df_combined = pi0_pos_df_combined.iloc[shuffled_idx].reset_index(drop=True)

    shuffled_idx = np.random.permutation(len(pi0_neg_df_combined))
    pi0_neg_df_combined = pi0_neg_df_combined.iloc[shuffled_idx].reset_index(drop=True)

    #
    shuffled_idx = np.random.permutation(len(kine_all_df_combined))
    kine_all_df_combined = kine_all_df_combined.iloc[shuffled_idx].reset_index(drop=True)

    shuffled_idx = np.random.permutation(len(kine_pos_df_combined))
    kine_pos_df_combined = kine_pos_df_combined.iloc[shuffled_idx].reset_index(drop=True)

    shuffled_idx = np.random.permutation(len(kine_neg_df_combined))
    kine_neg_df_combined = kine_neg_df_combined.iloc[shuffled_idx].reset_index(drop=True)

    # Verify alignment is preserved
    print(f"\n First few feature-label pairs:")
    for i in range(5):
        print(f"  Row {i}: X shape {X_train_combined.iloc[i:i+1].shape}, y = {y_train_combined.iloc[i]}")
        
    # Save combined dataset
    joblib.dump(X_train_combined, f'{data_dir}/X_train_combined.pkl')
    joblib.dump(y_train_combined, f'{data_dir}/y_train_combined.pkl')
    
    joblib.dump(X_val_combined, f'{data_dir}/X_val_combined.pkl')
    joblib.dump(y_val_combined, f'{data_dir}/y_val_combined.pkl')

    joblib.dump(X_test_combined, f'{data_dir}/X_test_combined.pkl')
    joblib.dump(y_test_combined, f'{data_dir}/y_test_combined.pkl')

    joblib.dump(all_df_combined, f'{data_dir}/all_df_combined.pkl')
    joblib.dump(pos_df_combined, f'{data_dir}/pos_df_combined.pkl')
    joblib.dump(neg_df_combined, f'{data_dir}/neg_df_combined.pkl')

    joblib.dump(pi0_all_df_combined, f'{data_dir}/pi0_all_df_combined.pkl')
    joblib.dump(pi0_pos_df_combined, f'{data_dir}/pi0_pos_df_combined.pkl')
    joblib.dump(pi0_neg_df_combined, f'{data_dir}/pi0_neg_df_combined.pkl')

    joblib.dump(kine_all_df_combined, f'{data_dir}/kine_all_df_combined.pkl')
    joblib.dump(kine_pos_df_combined, f'{data_dir}/kine_pos_df_combined.pkl')
    joblib.dump(kine_neg_df_combined, f'{data_dir}/kine_neg_df_combined.pkl')
    '''
