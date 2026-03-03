
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import joblib
import uproot
import numpy as np
import pandas as pd
from plots import plot_compr_hist, plot_var, plot_feature_pairs, plot_feature_target
from training.config import prepare_3photon_paris
from sklearn.model_selection import train_test_split
import random
import awkward as ak

def create_dataset(df): # For photon 4-momentum

    # Define photon 4-momentum
    br_nm = ['Br_E1', 'Br_px1', 'Br_py1', 'Br_pz1', 
             'Br_E2', 'Br_px2', 'Br_py2', 'Br_pz2', 
             'Br_E3', 'Br_px3', 'Br_py3', 'Br_pz3',
             'Br_m3pi', 'Br_lagvalue_min_7C', 'Br_deltaE',
             'Br_angle_pi0gam12', 'Br_ppIM', 'Br_betapi0',
             'Br_recon_indx', 'Br_bkg_indx']
    
    # Selection cut
    df = df[(df['Br_lagvalue_min_7C'] < 100)][br_nm]

    # Create all_df, pos_df, neg_df for signal and background events
    if len(br_nm): # Check para length and br_nm length are consistent
        
        if info['category'] == 'signal':
            print(f"Creating all_df for {info['category']} {df.columns}...")

            pos_df = df[(df['Br_recon_indx'] == 2) & (df['Br_bkg_indx'])][br_nm]
            neg_df = df[~((df['Br_recon_indx'] == 2) & (df['Br_bkg_indx']))][br_nm]

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

            # Combine pos + neg dataset and shuffling
            all_df = pd.concat([pos_df, neg_df], ignore_index=True)
            all_df = all_df.sample(frac=1).reset_index(drop=True)
      
        elif info['category'] == 'background':
            print(f"Creating pho4mom_all_df for {info['category']} {df.columns} ...")

            all_df = df

            nb_all_df = [i for i in range(len(all_df))]  
            all_df.insert(0, 'event', nb_all_df)  # Add event column

            is_signal_indx = [0] * len(all_df) 
            all_df['is_signal'] = is_signal_indx # Add is_signal column

            true_pi0_index  = [(-1, -1)] * len(all_df)
            all_df['true_pi0_pair'] = true_pi0_index # Add true_pi0_pair column
        else:
            print("No dataset other than signal or bkg is expected!")
    else:

        raise ValueError("Array length mismatch - cannot proceed")

    # pi0 features for ML learning
    pi0_all_df = prepare_3photon_paris(all_df)
    return all_df, pi0_all_df

##
def data_splitting(all_df):
    print(f'\n✅ Splitting dataset ...')

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

    #print("CREATING PAIRS FROM EACH SPLIT")

    pair_train = prepare_3photon_paris(all_df_train)
    pair_val = prepare_3photon_paris(all_df_val)
    pair_test = prepare_3photon_paris(all_df_test)

    #print(f"Training pairs:   {len(pair_train)}")
    #print(f"Validation pairs: {len(pair_val)}")
    #print(f"Test pairs:       {len(pair_test)}")

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

   

if __name__ == '__main__':
    #============================================================
    # LOAD INPUT ROOT FILES
    #============================================================
    f_nm = "../data/kloe_small_sample.root"

    # Create ooutput directory
    data_dir = rf'./dataset' 
    plot_dir = rf'./plots'
        
    os.makedirs(data_dir , exist_ok=True)
    os.makedirs(plot_dir , exist_ok=True)

    ## Loop over branches and create phys_map dynamically
    try:
        # Open the root file
        root_file = uproot.open(f_nm)
        branches = root_file.keys()
        print('All keys:', branches)
    
    except Exception as e:
        print('Error opening file: ', e)

    # Check first few braches and create a phys_map dynamically
    phys_map = {}

    for i, br_nm in enumerate(branches):
        # Remove ROOT cycle number (;1, ;2, etc.) for comparison
        base_br_nm = br_nm.split(';')[0]
        #print(base_br_nm)
        if base_br_nm == "TISR3PI_SIG":
            br_title = rf"$e^{{+}}e^{{-}}\to\pi^{{+}}\pi^{{-}}\pi^{{0}}\gamma$"
            category = "signal"
        #elif base_br_nm == "TETAGAM":
        #    br_title = rf"$e^{{+}}e^{{-}}\to\phi\to\eta\gamma$"
        #    category = "signal"
        #elif base_br_nm == "TKSL":
        #    br_title = rf"$e^{{+}}e^{{-}}\to\phi\to K_{{S}}K_{{L}}$"
        #    category = "signal"
        else:
            #br_title = "br_title"
            #category = "rest"
            continue

        #print(i, br_nm)

        # Fill phys_map entries
        phys_map[base_br_nm] = {
            #'br_nm': base_br_nm,
            'br_title': br_title,
            'category': category
        }

    # Add a single new entry
    #phys_map['NEW_BRANCH_NAME'] = {
    #'br_nm': 'NEW_BRANCH_NAME',
    #'br_title': r"$e^{+}e^{-}\to \text{Your Process}$",
    #   'category': 'signal'  # or 'background', 'rest', etc.
    #}

    #print(phys_map)

    # Save phys_map
    joblib.dump(phys_map, f'{data_dir}/phys_map.pkl')
    
    # Check phys_map
    for data_type, info in phys_map.items():
        info_title = info['br_title']
        info_category = info['category']

        print(data_type, info_title, info_category)

        if info_category == 'signal':
            print(f"Processing: {data_type}")
            tree = root_file[data_type]
            print(f"Tree entries: {tree.num_entries}")

            # Read as akward array (more flexible)
            # Exclude arra fields which confuse the pd expancding
            ak_array = tree.arrays(library="ak")

            #if "Br_m3pi" not in ak_array.fields:
            #    print("Br_m3pi is missing !!! Need to add this branch into the root file !!!")
            if "Br_mpi0" not in ak_array.fields:
                print("Br_mpi0 is missing !!! Need to add this branch into the root file !!!")
            
            

            print(f"\nAwkward array fields: {ak_array.fields}")
            print(f"Number of fields: {len(ak_array.fields)}")
            exclude_fields = ['Br_pull_E1', 'Br_pull_x1', 'Br_pull_y1', 'Br_pull_z1', 'Br_pull_t1']
            fields_to_use = []
            for field in ak_array.fields:
                # Skip fields that match exclude_fields
                if any(pattern in field for pattern in exclude_fields):
                    print(f"Excluding array field: {field}")
                    continue

                # Only include 1D fields
                if ak_array[field].ndim == 1:
                    fields_to_use.append(field)
                else:
                    print(f"Excluding multi-dim field: {field} (ndim={ak_array[field].ndim})")

        print(len(fields_to_use))
        #df = tree.arrays(library="pd") 
        #print(df.describe())


    #============================================================
    # CREATE DATASET
    #============================================================
    all_df_list = [] # List for storing all dataset

    for data_type, info in phys_map.items():
        #info_br = info['br_nm']
        br_title = info['br_title']
        
        ##
        data_nm = data_type.split(';')[0]
        if (data_nm == "TISR3PI_SIG"):
        #if (data_type == "TISR3PI_SIG"):

            print("="*50)
            # Create data frame
            tree = root_file[data_type] 
            df = tree.arrays(fields_to_use, library="pd") 
            #print(df.describe())

            # Check for any missing values
            print(f"\nMissing values per column:")
            print(df.isnull().sum())

            # Create pho4mom_all_df for signal and bkg separately
            all_df, pi0_all_df = create_dataset(df)

            # Data splitting
            all_df_train, all_df_val, all_df_test, X_train, y_train, X_val, y_val, X_test, y_test = data_splitting(all_df)

            joblib.dump(all_df, f'{data_dir}/all_df_{data_nm}.pkl')
            joblib.dump(pi0_all_df, f'{data_dir}/pi0_all_df_{data_nm}.pkl')

            joblib.dump(all_df_train, f'{data_dir}/all_df_train_{data_nm}.pkl')
            joblib.dump(all_df_val, f'{data_dir}/all_df_val_{data_nm}.pkl')
            joblib.dump(all_df_test, f'{data_dir}/all_df_test_{data_nm}.pkl')

            joblib.dump(X_train, f'{data_dir}/X_train_{data_nm}.pkl')
            joblib.dump(X_val, f'{data_dir}/X_val_{data_nm}.pkl')
            joblib.dump(X_test, f'{data_dir}/X_test_{data_nm}.pkl')

            joblib.dump(y_train, f'{data_dir}/y_train_{data_nm}.pkl')
            joblib.dump(y_val, f'{data_dir}/y_val_{data_nm}.pkl')
            joblib.dump(y_test, f'{data_dir}/y_test_{data_nm}.pkl')

            # Combine dataset
            all_df_list.append(all_df)
        else:
            continue

        ##
        #if all_df_list:
        #    all_df_comb = pd.concat(all_df_list, ignore_index=True)
        #    shuffled_idx = np.random.permutation(len(all_df_comb))
        #    all_df_comb = all_df_comb.iloc[shuffled_idx].reset_index(drop=True)
            
        #    joblib.dump(all_df_comb, f'{data_dir}/all_df_comb.pkl')


    # shuffle comb. dataset, using small sample,
    # split comb. dataset
    # save the comb. data set
    r'''
    # Combining dataset 
    # (later, after signal and background are separated)
    if df_list:
        df_combined = pd.concat(df_list, ignore_index=True)

        # Shuffle together using the same index
        shuffled_idx = np.random.permutation(len(df_combined))
        df_combined = df_combined.iloc[shuffled_idx].reset_index(drop=True)
            
        print(f"\nCombined DataFrame shape: {df_combined.shape}")
        print(f"Total entries: {len(df_combined)}")
        print(f"Columns: {df_combined.columns.tolist()}")

        # Check for any missing values
        print(f"\nMissing values per column:")
        #print(df_combined.isnull().sum())
    else:
        print("No dataframes were created successfully")
    '''
    #print(df_list[0].describe())