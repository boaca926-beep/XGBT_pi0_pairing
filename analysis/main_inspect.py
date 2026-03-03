import joblib
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from plots import plot_compr_hist, plot_var, plot_feature_pairs, plot_feature_target
import matplotlib.pyplot as plt
import numpy as np


# Inspect features and correlations

if __name__ == '__main__':

    #============================================================
    # LOAD INPUT DATA
    #============================================================
    input_data_dir = os.path.join(project_root, f'analysis/dataset')
    phys_map = joblib.load(os.path.join(input_data_dir, f'phys_map.pkl'))
    #print(phys_map)

    phys_ch = ['TISR3PI_SIG', 'signal']

    # Create output folder
    plot_dir = rf'./plots'
    os.makedirs(plot_dir , exist_ok=True)
    
    for data_type, info in phys_map.items():
        br_nm = data_type.split(';')[0]
        br_title = info['br_title']
        category = info['category']
        print(f"Inspecting dataset {br_nm}; {br_title}; {category}")  
        #print(info)

        if (br_nm  == phys_ch[0]and category == phys_ch[1]):
            print(f'{info}')

            ## Load dataset
            all_df = joblib.load(os.path.join(input_data_dir, f'all_df_{data_type}.pkl'))
            pos_df = all_df[all_df['is_signal'] == 1]
            neg_df = all_df[all_df['is_signal'] == 0]   
            df_set = [all_df.drop(['event', 'Br_m3pi', 'Br_lagvalue_min_7C', 'Br_recon_indx', 'Br_bkg_indx', 'is_signal', 'true_pi0_pair'], axis=1), pos_df, neg_df] 
            #print(all_df.columns)
            #print(pos_df.head(5))
            #print(neg_df.head(5))

            pi0_all_df = joblib.load(os.path.join(input_data_dir, f'pi0_all_df_{data_type}.pkl'))
            pi0_pos_df = pi0_all_df[pi0_all_df['is_pi0'] == 1]
            pi0_neg_df = pi0_all_df[pi0_all_df['is_pi0'] == 0]
            #print(pi0_all_df.columns)
            pi0_mass = pi0_pos_df['m_gg'].tolist()

            kine_all_df = all_df[['Br_m3pi', 'Br_lagvalue_min_7C', 'is_signal']]
            kine_pos_df = kine_all_df[kine_all_df['is_signal'] == 1]
            kine_neg_df = kine_all_df[~(kine_all_df['is_signal'] == 1)]      

            ## 1. Plot gamma 4-momentum 
            fig_compr_hist = plot_compr_hist(df_set, 3, 100, rf"$\gamma$ 4-momentum ({br_title})") # Photon 4-momentum comparison plot
            fig_compr_hist.savefig(f'{plot_dir}/Photon_4-momentum_compr_{br_nm}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_compr_hist)
            #print(all_df.columns.tolist())

            ## 2. Plot true pi0 mass
            fig_var = plot_var(pi0_mass, 'm_gg', br_title)
            fig_var.savefig(f'./{plot_dir}/pos_pi0_mass_{br_nm}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_var)

            ## 3. Plot kine. var
            drop_columns = ['is_signal']
            kine_df_set = [kine_all_df.drop(drop_columns, axis=1),
                          kine_pos_df.drop(drop_columns, axis=1),
                          kine_neg_df.drop(drop_columns, axis=1)]
            fig_compr_hist = plot_compr_hist(kine_df_set, 1, 100, rf"Kinematic Variables ({br_title})") # Pi0 comparison plot
            fig_compr_hist.savefig(f'./{plot_dir}/Kine_compr_{br_nm}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_compr_hist)
            #print(kine_all_df.columns)

            # 4. Plot pi0 features
            drop_columns = ['event', 'pair_id', 'is_pi0']
            pi0_df_set = [pi0_all_df.drop(drop_columns, axis=1),
                          pi0_pos_df.drop(drop_columns, axis=1),
                          pi0_neg_df.drop(drop_columns, axis=1)]
            fig_compr_hist = plot_compr_hist(pi0_df_set, 2, 100, rf"$\pi^{0}$ Candidates ({br_title})") # Pi0 comparison plot
            fig_compr_hist.savefig(f'./{plot_dir}/Pi0_compr_{br_nm}.png', dpi=300, bbox_inches='tight')
            plt.close(fig_compr_hist)    

            # 5. Plot pi0 feature-feature correlations
            fig_feature_pairs = plot_feature_pairs(pi0_all_df, rf"$\pi^{0}$ Candidates Feature-feature (Signal=Blue, Background=Red) ({br_title})")
            #fig_feature_pairs.savefig(f'./{plot_dir}/FF_correlation_{br_nm}.png', dpi=300, bbox_inches='tight')
            #plt.close(fig_feature_pairs.fig)

            # 6. Plot pi0 feature-target correlations
            features = ['m_gg', 'opening_angle', 'cos_theta', 'E_asym', 'e_min_x_angle', 'E1', 'E2', 'E3', 'asym_x_angle', 'E_diff', 'is_pi0']
            target_corr = pi0_all_df[features].corr()['is_pi0'].drop('is_pi0') #.sort_values(ascending=False)
            sorted_by_abs = target_corr.abs().sort_values(ascending=False)
            #fig_feature_target = plot_feature_target(target_corr, rf'Feature Importance: Correlation with true $\pi^{0}$ ({br_title})')
            #fig_feature_target.savefig(f'./{plot_dir}/feature_target_correlation_{br_nm}.png', dpi=300, bbox_inches='tight')
            #plt.close(fig_feature_target)
      
