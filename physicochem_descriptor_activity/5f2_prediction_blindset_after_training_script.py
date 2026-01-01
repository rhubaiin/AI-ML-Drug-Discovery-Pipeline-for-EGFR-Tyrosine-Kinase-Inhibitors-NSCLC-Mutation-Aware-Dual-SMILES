import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model
from rdkit import Chem
import sys
import os

# Import your feature generation functions from the 5f script
from adv_physchem5f2 import (
    generate_lig_inter_features,
    generate_lig_intra_features,
    generate_mut_inter_features,
    generate_mut_intra_features,
    generate_custom_features,
    generate_inter_interaction_features,
    generate_intra_interaction_features,
    generate_final_interaction_features
)

def load_prediction_resources():
    """
    Loads all models and scalers once to avoid memory leaks.
    Specific to 5f architecture (Shared Feature Scalers + Dual Y Scalers).
    """
    print("\n" + "="*80)
    print("LOADING MODELS AND SCALERS (ONCE) [5f Version]")
    print("="*80)
    
    resources = {}
    
    # 1. Load Y Scalers (Activity & Docking)
    if os.path.exists('y_scalers.pkl'):
        with open('y_scalers.pkl', 'rb') as f:
            y_scalers = pickle.load(f)
            resources['y_scaler1'] = y_scalers['y_scaler1']  # Activity
            resources['y_scaler2'] = y_scalers['y_scaler2']  # Docking
        print("✓ Loaded y_scalers (Activity & Docking)")
    else:
        print("❌ ERROR: y_scalers.pkl not found")
        return None

    # 2. Load Feature Scalers (List of dists)
    if os.path.exists('feature_scalers.pkl'):
        with open('feature_scalers.pkl', 'rb') as f:
            # This is a list of scaler dicts, one per mutation site index
            resources['feature_scalers_list'] = pickle.load(f)
        print("✓ Loaded feature_scalers.pkl")
    else:
        print("❌ ERROR: feature_scalers.pkl not found")
        return None

    # 3. Load RNN model
    if os.path.exists('rnn_sequential_model.h5'):
        try:
            resources['rnn_model'] = load_model('rnn_sequential_model.h5')
            print("✓ Loaded RNN model")
        except Exception as e:
             print(f"❌ ERROR loading RNN model: {e}")
             return None
    else:
        print("❌ ERROR: rnn_sequential_model.h5 not found")
        return None
        
    resources['embedding_models'] = {}
    
    mutation_sites = ['FULL_SMILES', 'ATP_POCKET', 'P_LOOP_HINGE', 'C_HELIX', 'DFG_A_LOOP', 'HRD_CAT']
    
    # 4. Load Hierarchical Models
    for site_name in mutation_sites:
        model_path = f'hierarchical_model_{site_name}.h5'
        if os.path.exists(model_path):
            try:
                # Compile=False is safer if loss functions changed, 
                # but we just need weights and graph frozen
                full_model = load_model(model_path, compile=False)
                
                # Create embedding model immediately to extract 16-dim vector
                embedding_model = Model(
                    inputs=full_model.inputs,
                    outputs=full_model.get_layer('embedding_output').output
                )
                resources['embedding_models'][site_name] = embedding_model
                print(f"✓ Loaded {site_name} model & extractor")
            except Exception as e:
                print(f"❌ ERROR loading model {site_name}: {e}")
                return None
        else:
             print(f"❌ ERROR: {model_path} not found")
             return None
             
    return resources

def predict_for_mutation_profile(ligand_names, ligand_smiles, mutation_profile_smiles, mutation_name, resources):

    print(f"\n{'='*80}")
    print(f"Processing Mutation: {mutation_name}")
    print(f"{'='*80}")
    
    y_scaler1 = resources['y_scaler1'] # Activity
    y_scaler2 = resources['y_scaler2'] # Docking
    feature_scalers_list = resources['feature_scalers_list'] # List of dicts
    
    mutation_sites = [
        ('FULL_SMILES', mutation_profile_smiles['smiles_full_egfr']),
        ('ATP_POCKET', mutation_profile_smiles['smiles_atp_pocket']),
        ('P_LOOP_HINGE', mutation_profile_smiles['smiles_p_loop']),
        ('C_HELIX', mutation_profile_smiles['smiles_c_helix']),
        ('DFG_A_LOOP', mutation_profile_smiles['smiles_dfg_a_loop']),
        ('HRD_CAT', mutation_profile_smiles['smiles_catalytic_hrd_motif'])
    ]
    
    if any(pd.isna(smi) for _, smi in mutation_sites):
        print(f"  ⚠️ WARNING: Missing SMILES for {mutation_name}, skipping...")
        return None
    
    ligand_embeddings_all_sites = []
    valid_ligand_indices = None
    
    for site_idx, (site_name, mut_smi) in enumerate(mutation_sites):
        print(f"  Processing {site_name}...")
        
        mut_inter = generate_mut_inter_features(mut_smi)
        mut_intra = generate_mut_intra_features(mut_smi)
        
        if mut_inter is None or mut_intra is None:
            print(f"    ❌ ERROR: Failed to generate mutation features for {site_name}")
            return None
        
        # Get correct scaler dict for this site index
        scalers = feature_scalers_list[site_idx]
        
        ligand_features = {
            'lig_inter': [], 'mut_inter': [], 'inter_interaction': [],
            'lig_intra': [], 'mut_intra': [], 'intra_interaction': [],
            'lig_mut_mix_inter_intra': [], 'final_fp_interaction': []
        }
        site_valid_indices = []
        
        for idx, lig_smi in enumerate(ligand_smiles):
            lig_inter = generate_lig_inter_features(lig_smi)
            lig_intra = generate_lig_intra_features(lig_smi)
            
            if lig_inter is None or lig_intra is None:
                continue
            
            lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra = generate_custom_features(
                lig_inter, mut_inter, lig_intra, mut_intra
            )
            
            inter_interaction = generate_inter_interaction_features(lig_inter, mut_inter)
            intra_interaction = generate_intra_interaction_features(lig_intra, mut_intra)
            
            if len(lig_mut_inter) > 0:
                inter_interaction = np.concatenate([np.array(lig_mut_inter), inter_interaction])
            if len(lig_mut_intra) > 0:
                intra_interaction = np.concatenate([np.array(lig_mut_intra), intra_interaction])
            
            final_fp_interaction = generate_final_interaction_features(lig_smi, mut_smi)
            
            ligand_features['lig_inter'].append(lig_inter)
            ligand_features['mut_inter'].append(mut_inter)
            ligand_features['inter_interaction'].append(inter_interaction)
            ligand_features['lig_intra'].append(lig_intra)
            ligand_features['mut_intra'].append(mut_intra)
            ligand_features['intra_interaction'].append(intra_interaction)
            ligand_features['lig_mut_mix_inter_intra'].append(np.array(lig_mut_mix_inter_intra))
            ligand_features['final_fp_interaction'].append(final_fp_interaction)
            site_valid_indices.append(idx)
        
        if len(site_valid_indices) == 0:
            print(f"    ❌ ERROR: No valid ligands for {site_name}")
            return None
        
        if valid_ligand_indices is None:
            valid_ligand_indices = site_valid_indices
        
        # Scale features
        scaled_features = {}
        for key in ligand_features.keys():
            arr = np.array(ligand_features[key])
            scaled_features[key] = scalers[key].transform(arr)
        
        embedding_model = resources['embedding_models'][site_name]
        
        site_embeddings = embedding_model.predict([
            scaled_features['final_fp_interaction'],
            scaled_features['lig_mut_mix_inter_intra'],
            scaled_features['inter_interaction'],
            scaled_features['intra_interaction'],
            scaled_features['mut_inter'],
            scaled_features['lig_inter'],
            scaled_features['mut_intra'],
            scaled_features['lig_intra']
        ], verbose=0)
        
        ligand_embeddings_all_sites.append(site_embeddings)
    
    if len(ligand_embeddings_all_sites) != 6:
        print(f"  ❌ ERROR: Incomplete embeddings (got {len(ligand_embeddings_all_sites)}/6)")
        return None
    
    # Check architecture flow
    sequential_embeddings = np.stack(ligand_embeddings_all_sites, axis=1)
    
    rnn_model = resources['rnn_model']
    # RNN returns list of [activity, docking]
    predictions = rnn_model.predict(sequential_embeddings, verbose=0)
    
    pred_activity = predictions[0].flatten()
    pred_docking = predictions[1].flatten()
    
    # 1. Activity: Inverse Transform + Expm1 (because training used log1p)
    pred_activity = y_scaler1.inverse_transform(pred_activity.reshape(-1, 1)).flatten()
    pred_activity = np.expm1(pred_activity)
    
    # 2. Docking: Inverse Transform only (training used raw standardization)
    pred_docking = y_scaler2.inverse_transform(pred_docking.reshape(-1, 1)).flatten()
    
    print(f"  ✓ Generated {len(pred_activity)} predictions")
    print(f"    Mean Activity: {pred_activity.mean():.2f}")
    print(f"    Mean Docking: {pred_docking.mean():.2f}")
    
    results_df = pd.DataFrame({
        'mutation_name': [mutation_name] * len(pred_activity),
        'ligand_name': ligand_names[valid_ligand_indices],
        'ligand_smiles': ligand_smiles.iloc[valid_ligand_indices].values,
        'predicted_activity': pred_activity,
        'predicted_docking': pred_docking 
    })
    
    return results_df


def predict_all_mutations(ligand_csv_path, mutation_profiles_csv_path=None):

    print("\n" + "="*80)
    print("AUTO-PREDICTION FOR ALL MUTATIONS [5f Dual-Output Version]")
    print("="*80)
    
    resources = load_prediction_resources()
    if resources is None:
        print("❌ CRITICAL ERROR: Failed to load models/scalers. Exiting.")
        sys.exit(1)

    print(f"\nLoading ligands from: {ligand_csv_path}")
    df_ligands = pd.read_csv(ligand_csv_path)
    
    # Grab ligand names from first column
    ligand_names = df_ligands.iloc[:, 0].values
    
    # Grab SMILES
    if 'smiles' in df_ligands.columns:
        ligand_smiles = df_ligands['smiles']
    elif df_ligands.shape[1] >= 2:
        print("  Note: 'smiles' column not found, using 2nd column as SMILES.")
        ligand_smiles = df_ligands.iloc[:, 1]
    else:
        print(f"❌ ERROR: CSV must have 'smiles' column or at least 2 columns!")
        sys.exit(1)
    print(f"✓ Loaded {len(ligand_smiles)} ligands")
    
    # --- Mutation Loading Logic (Same as before) ---
    if mutation_profiles_csv_path is None:
        print("\nSearching for training data in current directory...")
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        possible_names = [
            'df_combo_tki_ed_smiles_lig_activity_validated.csv', 
            'training_data.csv', 'df_train.csv'
        ]
        for csv_file in csv_files:
            if 'validated' in csv_file.lower() or 'train' in csv_file.lower():
                if csv_file not in possible_names: possible_names.insert(0, csv_file)
        
        df_train = None
        for filename in possible_names:
            if filename in csv_files:
                try:
                    df_train = pd.read_csv(filename)
                    print(f"✓ Loaded training data from: {filename}")
                    break
                except: continue
        if df_train is not None: df_train.columns = df_train.columns.str.strip()
    else:
        df_train = pd.read_csv(mutation_profiles_csv_path)
        df_train.columns = df_train.columns.str.strip()
    
    if df_train is None:
        print("❌ ERROR: Could not find any mutation profile data.")
        sys.exit(1)

    mutation_profile_columns = [
        'smiles_full_egfr', 'smiles 718_862_atp_pocket', 'smiles_p_loop',
        'smiles_c_helix', 'smiles_l858r_a_loop_dfg_motif', 'smiles_catalytic_hrd_motif',
        'tkd'
    ]
    
    unique_mutation_profiles = df_train[mutation_profile_columns].drop_duplicates().reset_index(drop=True)
    print(f"✓ Found {len(unique_mutation_profiles)} unique mutation profiles")
    
    # Process
    all_results = []
    
    for profile_idx, profile_row in unique_mutation_profiles.iterrows():
        mutation_name = profile_row['tkd']
        mutation_profile_smiles = {
            'smiles_full_egfr': profile_row['smiles_full_egfr'],
            'smiles_atp_pocket': profile_row['smiles 718_862_atp_pocket'],
            'smiles_p_loop': profile_row['smiles_p_loop'],
            'smiles_c_helix': profile_row['smiles_c_helix'],
            'smiles_dfg_a_loop': profile_row['smiles_l858r_a_loop_dfg_motif'],
            'smiles_catalytic_hrd_motif': profile_row['smiles_catalytic_hrd_motif']
        }
        
        results_df = predict_for_mutation_profile(
            ligand_names,
            ligand_smiles, 
            mutation_profile_smiles, 
            mutation_name,
            resources
        )
        
        if results_df is not None:
            all_results.append(results_df)
    
    if len(all_results) == 0:
        print("\n❌ ERROR: No predictions were generated!")
        sys.exit(1)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    output_file = 'predictions_5f_dual_output.csv'
    combined_results.to_csv(output_file, index=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    return combined_results

if __name__ == "__main__":
    ligand_csv = '/mnt/d/Publications/project_insilico/activity_physicochem_descriptor/preds/adv_physchem5f_all_lig_full_smiles/placebo_blind_set1_for_predicitions.csv'
    mutation_csv = None
    if len(sys.argv) > 1: ligand_csv = sys.argv[1]
    if len(sys.argv) > 2: mutation_csv = sys.argv[2]
    
    predict_all_mutations(ligand_csv, mutation_csv)
