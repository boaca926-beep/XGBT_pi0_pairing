
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

def create_dataset(df, category): # For photon 4-momentum
    print(f'\n✅ Creating dataset ...')
    # Define photon 4-momentum
    br_nm = ['Br_E1', 'Br_px1', 'Br_py1', 'Br_pz1', 
             'Br_E2', 'Br_px2', 'Br_py2', 'Br_pz2', 
             'Br_E3', 'Br_px3', 'Br_py3', 'Br_pz3',
             'Br_m3pi', 'Br_lagvalue_min_7C', 'Br_deltaE',
             'Br_angle_pi0gam12', 'Br_ppIM', 'Br_betapi0',
             'Br_recon_indx', 'Br_bkg_indx']
    
    # Selection cut, ensure physical region
    cut_region = (df['Br_lagvalue_min_7C'] < 100) # Selection cuts
    phys_region = (df['Br_betapi0'] < 1) & (df['Br_betapi0'] > 0) # Physical region

    #df = df[(df['Br_lagvalue_min_7C'] < 100) & (df['Br_betapi0'] < 1) & (df['Br_betapi0'] > 0)][br_nm]
    df = df[cut_region & phys_region][br_nm]

    # Create all_df, pos_df, neg_df for signal and background events
    if len(br_nm): # Check para length and br_nm length are consistent
        
        if category == 'signal':
            print(f"Creating all_df for {category} {df.columns}...")

            pos_df = df[(df['Br_recon_indx'] == 2) & (df['Br_bkg_indx'] == 1)][br_nm]
            neg_df = df[~((df['Br_recon_indx'] == 2) & (df['Br_bkg_indx'] == 1))][br_nm]

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
      
        elif category == 'background':
            print(f"Creating pho4mom_all_df for {category} {df.columns} ...")

            all_df = df

            nb_all_df = [i for i in range(len(all_df))]  
            all_df.insert(0, 'event', nb_all_df)  # Add event column

            is_signal_indx = [0] * len(all_df) 
            all_df['is_signal'] = is_signal_indx # Add is_signal column

            true_pi0_index  = [(-1, -1)] * len(all_df)
            all_df['true_pi0_pair'] = true_pi0_index # Add true_pi0_pair column
        else: # combined or others
            raise ValueError("Only sig and bkg allow!")
    else: 
        print("No dataset other than signal or bkg or combined is expected!")
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

    # Verify event column preservation
    for name, pair_df in [('Train', pair_train), ('Val', pair_val), ('Test', pair_test)]:
        if 'event' not in pair_df.columns:
            print(f"❌ CRITICAL: 'event' column missing in {name} pairs!")
            print(f"   Columns: {pair_df.columns.tolist()}")
        else:
            n_events = pair_df['event'].nunique()
            n_pairs = len(pair_df)
            print(f"✅ {name}: {n_pairs} pairs from {n_events} events ({n_pairs/n_events:.2f} pairs/event)")

    # Verify event column preservation
    for name, pair_df in [('Train', pair_train), ('Val', pair_val), ('Test', pair_test)]:
        if 'event' not in pair_df.columns:
            print(f"❌ CRITICAL: 'event' column missing in {name} pairs!")
            print(f"   Columns: {pair_df.columns.tolist()}")
        else:
            n_events = pair_df['event'].nunique()
            n_pairs = len(pair_df)
            print(f"✅ {name}: {n_pairs} pairs from {n_events} events ({n_pairs/n_events:.2f} pairs/event)")

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
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"  X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    return all_df_train, all_df_val, all_df_test, X_train, y_train, X_val, y_val, X_test, y_test, pair_train, pair_val, pair_test
    return all_df_train, all_df_val, all_df_test, X_train, y_train, X_val, y_val, X_test, y_test, pair_train, pair_val, pair_test

   

if __name__ == '__main__':
    #============================================================
    # LOAD INPUT ROOT FILES
    #============================================================
    #f_nm = "../data/kloe_sample.root"
    #f_nm = "../data/kloe_sample_chain.root"
    f_nm = "../data/kloe_sample_full.root"


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
    #All keys: ['TISR3PI_SIG;1', (done) 
    #           'TOMEGAPI;1', (done) 
    #           'TKPM;1', (done)
    #           'TKSL;1', (done) 
    #           'T3PIGAM;1', (no)
    #           'TRHOPI;1', (skipped if too few events)
    #           'TETAGAM;1', (done)
    #           'TBKGREST;1', (done)
    #           'TDATA;1', (no)
    #           'TEEG;1']

    for i, br_nm in enumerate(branches):
        # Remove ROOT cycle number (;1, ;2, etc.) for comparison
        base_br_nm = br_nm.split(';')[0]
        #print(base_br_nm)
        if base_br_nm == "TISR3PI_SIG":
            br_title = rf"$e^{{+}}e^{{-}}\to\pi^{{+}}\pi^{{-}}\pi^{{0}}\gamma$"
            category = "signal"
        elif base_br_nm == "TOMEGAPI":
            br_title = rf"$\omega\pi^{0}$"
            category = "background"
        elif base_br_nm == "TKPM":
            br_title = rf"$e^{{+}}e^{{-}}\to\phi\to K\bar{{K}}$"
            category = "background"
        elif base_br_nm == "TKSL":
            br_title = rf"$e^{{+}}e^{{-}}\to\phi\to K_{{S}}K_{{L}}$"
            category = "background"
        elif base_br_nm == "TRHOPI":
            br_title = rf"$e^{{+}}e^{{-}}\to\phi\to \rho\pi$"
            category = "background"
        elif base_br_nm == "TBKGREST":
            br_title = rf"Others"
            category = "background"
        elif base_br_nm == "TEEG":
            br_title = rf"$e^{{+}}e^{{-}}\to\phi\to e^{{+}}e^{{-}}\gamma$"
            category = "background"
        elif base_br_nm == "TETAGAM":
            br_title = rf"$e^{{+}}e^{{-}}\to\phi\to\eta\gamma$"
            category = "signal"

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
    
    # Check phys_map
    for data_type, info in phys_map.items():
        info_title = info['br_title']
        info_category = info['category']

        print(data_type, info_title, info_category)

        if info_category == 'signal':
            print(f"Creating phys_ch: {data_type}")
            tree = root_file[data_type]
            print(f"Tree entries: {tree.num_entries}")

            # Read as akward array (more flexible)
            # Exclude arra fields which confuse the pd expancding
            ak_array = tree.arrays(library="ak")

            #if "Br_m3pi" not in ak_array.fields:
            #    print("Br_m3pi is missing !!! Need to add this branch into the root file !!!")
            if "Br_mpi0" not in ak_array.fields:
                print("Br_mpi0 is missing !!! Need to add this branch into the root file !!!")
            
            #print(f"\nAwkward array fields: {ak_array.fields}")
            #print(f"Number of fields: {len(ak_array.fields)}")
            exclude_fields = ['Br_pull_E1', 'Br_pull_x1', 'Br_pull_y1', 'Br_pull_z1', 'Br_pull_t1']
            fields_to_use = []
            for field in ak_array.fields:
                # Skip fields that match exclude_fields
                if any(pattern in field for pattern in exclude_fields):
                    #print(f"Excluding array field: {field}")
                    continue

                # Only include 1D fields
                if ak_array[field].ndim == 1:
                    fields_to_use.append(field)
                else:
                    print(f"Excluding multi-dim field: {field} (ndim={ak_array[field].ndim})")

        #print(len(fields_to_use))
        #df = tree.arrays(library="pd") 
        #print(df.describe())


    #============================================================
    # CREATE DATASET
    #============================================================
    df_list = [] # List for storing all dataset for combining
    #fields_to_use = None
    #all_fields_collected = False
    #fields_to_use = None
    #all_fields_collected = False

    ch_indx = 0
    for data_type, info in phys_map.items():
        try:
            data_nm = data_type.split(';')[0]
            #if (data_nm == "TETAGAM"):
            #if (data_type == "TISR3PI_SIG"):
            #if (data_type == "TKSL"):
            #if (data_type == "TOMEGAPI"):

            ch_indx += 1
            print("="*25 + f"Channel {ch_indx}: {data_type}" + "="*25)
            
            # Get tree for this channel
            tree = root_file[data_type] 

            # Debug info for this channel
            print(f"Processing {data_type}...")
            print(f"Number of entries: {tree.num_entries}")

            # Check for any missing values
            #print(f"Missing values per column:")
            #print(df.isnull().sum())

            # Read as akward array (more flexible) to determine fields, only first 100 to save memory
            # Exclude arra fields which confuse the pd expancding
            ak_array = tree.arrays(library="ak", entry_stop=100)

            # Check for required branches
            if "Br_mpi0" not in ak_array.fields:
                print("Br_mpi0 is missing !!! Need to add this branch into the root file !!!")
            
            #print(f"\nAwkward array fields: {ak_array.fields}")
            #print(f"Number of fields: {len(ak_array.fields)}")

            # Generate fields_to_use for THIS channel
            exclude_fields = ['Br_pull_E1', 'Br_pull_x1', 'Br_pull_y1', 'Br_pull_z1', 'Br_pull_t1']
            fields_to_use = []

            for field in ak_array.fields:
                # Skip fields that match exclude_fields
                if any(pattern in field for pattern in exclude_fields):
                    #print(f"Excluding array field: {field}")
                    continue

                # Only include 1D fields
                if ak_array[field].ndim == 1:
                    fields_to_use.append(field)
                else:
                    print(f"Excluding multi-dim field: {field} (ndim={ak_array[field].ndim})")

            # Check if we have the required branches for create_dataset
            required_br = ['Br_E1', 'Br_px1', 'Br_py1', 'Br_pz1', 
                        'Br_E2', 'Br_px2', 'Br_py2', 'Br_pz2', 
                        'Br_E3', 'Br_px3', 'Br_py3', 'Br_pz3',
                        'Br_m3pi', 'Br_lagvalue_min_7C', 'Br_deltaE',
                        'Br_angle_pi0gam12', 'Br_ppIM', 'Br_betapi0',
                        'Br_recon_indx', 'Br_bkg_indx']
            
            missing_br = [br for br in required_br if br not in fields_to_use]
            if missing_br:
                print(f"WARNING: Missing branches in {data_type}: {missing_br}")
                if info['category'] == 'signal' and len(missing_br) > 5:
                    print(f"Too many missing branches for signal channel {data_type}, skipping...")
                    continue
            
            # Check a few events before full processing
            if len(fields_to_use) > 0:
                sample_fields = fields_to_use[:min(5, len(fields_to_use))]
                sample_df = tree.arrays(sample_fields, library="pd", entry_stop=100)
                print(f"Sample data shape: {sample_df.shape}")
                print(f"Sample columns: {sample_df.columns.tolist()}")
            
            print(f"Fields being used: {fields_to_use}")

            # Create full data frame
            print(f"Loading full dataset for {data_type}...")
            df = tree.arrays(fields_to_use, library="pd") 
            print(f"Full dataset shape: {df.shape}")

            # Create pho4mom_all_df for signal and bkg separately
            all_df, pi0_all_df = create_dataset(df, info['category'])
                
            # Check for anomalies
            #print(all_df.columns)
            betapi0_values = all_df['Br_betapi0']
            #print(betapi0_values.describe())

            # Data splitting
            if len(all_df) < 10:
                print("WARNING: Very few background events!")  
                continue

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

            # Combining dataset
            df_list.append(all_df) # Add each channel's dataframe
  
        except Exception as e:
            print(f"ERROR processing {data_type}: {str(e)}")
            import traceback
            traceback.print_exc()  # This will show you exactly what went wrong
            print(f"Skipping {data_type} and continuing with next channel...")
            continue

        

    ## Combining dataset
    if df_list: # combining dataset
        print(f"Combining {len(df_list)} channels ...")

        df_comb = pd.concat(df_list, ignore_index=True)
        print(f"Raw combined shape: {df_comb.shape}")

        # Shuffle
        df_comb = df_comb.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Shuffled combined shape: {df_comb.shape}")

        # Creat datasets
        all_df_comb = df_comb
        pi0_all_df_comb = prepare_3photon_paris(all_df_comb)  # This should work now
        #all_df_comb, pi0_all_df_comb = create_dataset(df_comb, 'combined')

        # Split
        all_df_train_comb, all_df_val_comb, all_df_test_comb, X_train_comb, y_train_comb, X_val_comb, y_val_comb, X_test_comb, y_test_comb, pair_train_comb, pair_val_comb, pair_test_comb = data_splitting(all_df_comb)
        all_df_train_comb, all_df_val_comb, all_df_test_comb, X_train_comb, y_train_comb, X_val_comb, y_val_comb, X_test_comb, y_test_comb, pair_train_comb, pair_val_comb, pair_test_comb = data_splitting(all_df_comb)

        # Save with clear names
        #save_dict = {
        #    'all_df_comb': all_df_comb,
        #    'pi0_all_df_comb': pi0_all_df_comb,
        #    'train': (all_df_train_comb, X_train_comb, y_train_comb),
        #    'val': (all_df_val_comb, X_val_comb, y_val_comb),
        #    'test': (all_df_test_comb, X_test_comb, y_test_comb)
        #}
        #save_dict = {
        #    'all_df_comb': all_df_comb,
        #    'pi0_all_df_comb': pi0_all_df_comb,
        #    'train': (all_df_train_comb, X_train_comb, y_train_comb),
        #    'val': (all_df_val_comb, X_val_comb, y_val_comb),
        #    'test': (all_df_test_comb, X_test_comb, y_test_comb)
        #}

        # Save individual files
        joblib.dump(all_df_comb, f'{data_dir}/all_df_TCOMB.pkl')
        joblib.dump(pi0_all_df_comb, f'{data_dir}/pi0_all_df_TCOMB.pkl')

        joblib.dump(all_df_train_comb, f'{data_dir}/all_df_train_TCOMB.pkl')
        joblib.dump(all_df_val_comb, f'{data_dir}/all_df_val_TCOMB.pkl')
        joblib.dump(all_df_test_comb, f'{data_dir}/all_df_test_TCOMB.pkl')

        # Save the pair DataFrames - THESE CONTAIN THE EVENT INFO!
        joblib.dump(pair_train_comb, f'{data_dir}/pair_train_TCOMB.pkl')
        joblib.dump(pair_val_comb, f'{data_dir}/pair_val_TCOMB.pkl')
        joblib.dump(pair_test_comb, f'{data_dir}/pair_test_TCOMB.pkl')

        # Save the pair DataFrames - THESE CONTAIN THE EVENT INFO!
        joblib.dump(pair_train_comb, f'{data_dir}/pair_train_TCOMB.pkl')
        joblib.dump(pair_val_comb, f'{data_dir}/pair_val_TCOMB.pkl')
        joblib.dump(pair_test_comb, f'{data_dir}/pair_test_TCOMB.pkl')

        joblib.dump(X_train_comb, f'{data_dir}/X_train_TCOMB.pkl')
        joblib.dump(X_val_comb, f'{data_dir}/X_val_TCOMB.pkl')
        joblib.dump(X_test_comb, f'{data_dir}/X_test_TCOMB.pkl')

        joblib.dump(y_train_comb, f'{data_dir}/y_train_TCOMB.pkl')
        joblib.dump(y_val_comb, f'{data_dir}/y_val_TCOMB.pkl')
        joblib.dump(y_test_comb, f'{data_dir}/y_test_TCOMB.pkl')  
        
        print(f"Combined data contains: {[k for k in phys_map.keys() if k != 'TDATA']}")  
    else:
        print(f"df_list is empty, no channels can be combined!")
        raise ValueError("No data to combine!")

    phys_map['TCOMB'] = {
        'br_title': "MC combined",
        'category': "combined"
    }
    print(phys_map.keys())

    # Save phys_map
    joblib.dump(phys_map, f'{data_dir}/phys_map.pkl')