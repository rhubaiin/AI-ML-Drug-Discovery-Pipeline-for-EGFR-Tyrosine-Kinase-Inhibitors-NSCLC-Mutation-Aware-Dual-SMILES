import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model
from rdkit import Chem
import sys
import os

# Import your feature generation functions
from adv_physchem5e2 import (
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
    Loads all models and scalers once to avoid memory leaks and performance issues.
    """
    print("\n" + "="*80)
    print("LOADING MODELS AND SCALERS (ONCE)")
    print("="*80)
    
    resources = {}
    
    # Load y_scaler
    if os.path.exists('y_scaler.pkl'):
        with open('y_scaler.pkl', 'rb') as f:
            resources['y_scaler'] = pickle.load(f)
        print("✓ Loaded y_scaler")
    else:
        print("❌ ERROR: y_scaler.pkl not found")
        return None

    # Load RNN model
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
        
    resources['scalers'] = {}
    resources['embedding_models'] = {}
    
    mutation_sites = ['FULL_SMILES', 'ATP_POCKET', 'P_LOOP_HINGE', 'C_HELIX', 'DFG_A_LOOP', 'HRD_CAT']
    
    # Load all scalers
    if os.path.exists('feature_scalers.pkl'):
        with open('feature_scalers.pkl', 'rb') as f:
            all_scalers = pickle.load(f)
        print("✓ Loaded feature_scalers.pkl")
    else:
        print("❌ ERROR: feature_scalers.pkl not found")
        return None

    # Verify scaler count
    if len(all_scalers) != 6:
        print(f"❌ ERROR: Expected 6 scaler sets, found {len(all_scalers)}")
        return None

    for i, site_name in enumerate(mutation_sites):
        # Assign Scaler
        resources['scalers'][site_name] = all_scalers[i]
            
        # Load Model and create Embedding Model
        model_path = f'hierarchical_model_{site_name}.h5'
        if os.path.exists(model_path):
            try:
                full_model = load_model(model_path)
                # Create embedding model immediately
                embedding_model = Model(
                    inputs=full_model.inputs,
                    outputs=full_model.get_layer('embedding_output').output
                )
                resources['embedding_models'][site_name] = embedding_model
                print(f"✓ Loaded {site_name} model & created embedding extractor")
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
    
    # Use pre-loaded y_scaler
    y_scaler = resources['y_scaler']
    
    mutation_sites = [
        ('FULL_SMILES', mutation_profile_smiles['smiles_full_egfr']),
        ('ATP_POCKET', mutation_profile_smiles['smiles_atp_pocket']),
        ('P_LOOP_HINGE', mutation_profile_smiles['smiles_p_loop']),
        ('C_HELIX', mutation_profile_smiles['smiles_c_helix']),
        ('DFG_A_LOOP', mutation_profile_smiles['smiles_dfg_a_loop']),
        ('HRD_CAT', mutation_profile_smiles['smiles_hrd_cat'])
    ]
    
    # Check for missing mutation SMILES
    if any(pd.isna(smi) for _, smi in mutation_sites):
        print(f"  ⚠️ WARNING: Missing SMILES for {mutation_name}, skipping...")
        return None
    
    ligand_embeddings_all_sites = []
    valid_ligand_indices = None
    
    for site_idx, (site_name, mut_smi) in enumerate(mutation_sites):
        print(f"  Processing {site_name}...")
        
        # Generate mutation features
        mut_inter = generate_mut_inter_features(mut_smi)
        mut_intra = generate_mut_intra_features(mut_smi)
        
        if mut_inter is None or mut_intra is None:
            print(f"    ❌ ERROR: Failed to generate mutation features for {site_name}")
            return None
        
        # Use pre-loaded scalers
        scalers = resources['scalers'][site_name]
        
        # Process each ligand
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
        
        # Store valid indices from first site
        if valid_ligand_indices is None:
            valid_ligand_indices = site_valid_indices
        
        # Scale features
        scaled_features = {}
        for key in ligand_features.keys():
            arr = np.array(ligand_features[key])
            scaled_features[key] = scalers[key].transform(arr)
        
        # Use pre-loaded embedding model
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
        print(f"    ✓ Extracted embeddings: {site_embeddings.shape}")
    
    # Stack embeddings and predict
    if len(ligand_embeddings_all_sites) != 6:
        print(f"  ❌ ERROR: Incomplete embeddings (got {len(ligand_embeddings_all_sites)}/6)")
        return None
    
    sequential_embeddings = np.stack(ligand_embeddings_all_sites, axis=1)
    print(f"\n  Stacked embeddings shape: {sequential_embeddings.shape}")
    
    # Use pre-loaded RNN model
    rnn_model = resources['rnn_model']
    predictions = rnn_model.predict(sequential_embeddings, verbose=0).flatten()
    
    # Inverse transform
    predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    predictions = np.expm1(predictions)
    
    print(f"  ✓ Generated {len(predictions)} predictions")
    print(f"    Range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"    Mean: {predictions.mean():.2f}")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'mutation_name': [mutation_name] * len(predictions),
        'ligand_name': ligand_names[valid_ligand_indices],
        'ligand_smiles': ligand_smiles.iloc[valid_ligand_indices].values,
        'predicted_activity': predictions
    })
    
    return results_df


def predict_all_mutations(ligand_csv_path, mutation_profiles_csv_path=None):

    print("\n" + "="*80)
    print("AUTO-PREDICTION FOR ALL MUTATIONS")
    print("="*80)
    
    # Load resources ONCE
    resources = load_prediction_resources()
    if resources is None:
        print("❌ CRITICAL ERROR: Failed to load models/scalers. Exiting.")
        sys.exit(1)

    # Load ligands
    print(f"\nLoading ligands from: {ligand_csv_path}")
    df_ligands = pd.read_csv(ligand_csv_path)
    
    # According to user: first column name of the ligand and the second column smiles
    # We will grab them by index to be safe as per "first column... second column"
    if df_ligands.shape[1] < 2:
        print(f"❌ ERROR: CSV must have at least 2 columns (Name, SMILES)!")
        sys.exit(1)

    ligand_names = df_ligands.iloc[:, 0].values
    ligand_smiles = df_ligands.iloc[:, 1]
    
    print(f"✓ Loaded {len(ligand_smiles)} ligands")
    print(f"  Sample Name: {ligand_names[0]}")
    print(f"  Sample SMILES: {ligand_smiles.iloc[0]}")
    
    # Load mutation profiles
    if mutation_profiles_csv_path is None:
        # Try to load from training data
        print("\nSearching for training data in current directory...")
        
        # List all CSV files in current directory
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        print(f"Found CSV files: {csv_files}")
        
        # Try multiple possible filenames
        possible_names = [
            'df_combo_tki_ed_smiles_lig_activity_validated.csv',
            'training_data.csv',
            'df_train.csv'
        ]
        
        # Also search for any file containing key terms
        for csv_file in csv_files:
            if 'validated' in csv_file.lower() or 'train' in csv_file.lower() or 'combo' in csv_file.lower():
                if csv_file not in possible_names:
                    possible_names.insert(0, csv_file)
        
        df_train = None
        for filename in possible_names:
            if filename in csv_files:
                try:
                    print(f"Trying to load: {filename}")
                    df_train = pd.read_csv(filename)
                    print(f"✓ Loaded training data from: {filename}")
                    print(f"  Shape: {df_train.shape}")
                    print(f"  Columns: {df_train.columns.tolist()[:5]}...")
                    break
                except Exception as e:
                    print(f"  Failed to load {filename}: {str(e)}")
                    continue
        
        if df_train is not None:
            df_train.columns = df_train.columns.str.strip()
    else:
        print(f"\nLoading mutation profiles from: {mutation_profiles_csv_path}")
        df_train = pd.read_csv(mutation_profiles_csv_path)
        df_train.columns = df_train.columns.str.strip()
    
    if df_train is None:
        print("❌ ERROR: Could not find any mutation profile data.")
        sys.exit(1)

    # Extract unique mutation profiles
    mutation_profile_columns = [
        'smiles_full_egfr',
        'smiles 718_862_atp_pocket',
        'smiles_p_loop',
        'smiles_c_helix',
        'smiles_l858r_a_loop_dfg_motif',
        'smiles_catalytic_hrd_motif',
        'tkd'
    ]
    
    # Check if columns exist
    missing_cols = [col for col in mutation_profile_columns if col not in df_train.columns]
    if missing_cols:
        print(f"❌ ERROR: Missing columns in mutation profiles: {missing_cols}")
        print(f"   Available columns: {df_train.columns.tolist()}")
        sys.exit(1)
    
    unique_mutation_profiles = df_train[mutation_profile_columns].drop_duplicates().reset_index(drop=True)
    print(f"✓ Found {len(unique_mutation_profiles)} unique mutation profiles")
    
    # Show mutation names
    print("\nMutation profiles to process:")
    for idx, row in unique_mutation_profiles.iterrows():
        print(f"  {idx+1}. {row['tkd']}")
    
    # Process each mutation profile
    all_results = []
    successful_predictions = 0
    failed_predictions = 0
    
    for profile_idx, profile_row in unique_mutation_profiles.iterrows():
        mutation_name = profile_row['tkd']
        
        # Prepare mutation profile dictionary
        mutation_profile_smiles = {
            'smiles_full_egfr': profile_row['smiles_full_egfr'],
            'smiles_atp_pocket': profile_row['smiles 718_862_atp_pocket'],
            'smiles_p_loop': profile_row['smiles_p_loop'],
            'smiles_c_helix': profile_row['smiles_c_helix'],
            'smiles_dfg_a_loop': profile_row['smiles_l858r_a_loop_dfg_motif'],
            'smiles_hrd_cat': profile_row['smiles_catalytic_hrd_motif']
        }
        
        # Predict for this mutation - PASS RESOURCES
        results_df = predict_for_mutation_profile(
            ligand_names,
            ligand_smiles, 
            mutation_profile_smiles, 
            mutation_name,
            resources
        )
        
        if results_df is not None:
            all_results.append(results_df)
            successful_predictions += 1
        else:
            failed_predictions += 1
            print(f"  ❌ Failed to generate predictions for {mutation_name}")
    
    # Combine all results
    if len(all_results) == 0:
        print("\n❌ ERROR: No predictions were generated!")
        sys.exit(1)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    output_file = 'predictions_5e2_on_blindset1.csv'
    combined_results.to_csv(output_file, index=False)
    
    # Summary statistics
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"Total ligands: {len(ligand_smiles)}")
    print(f"Total mutations: {len(unique_mutation_profiles)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"Total predictions generated: {len(combined_results)}")
    print(f"\n✓ Results saved to: {output_file}")
    
    return combined_results


if __name__ == "__main__":

    # Parse command line arguments
    ligand_csv = '/mnt/d/Publications/project_insilico/activity_physicochem_descriptor/preds/adv_physchem5e3_all_lig_full_smiles/placebo_blind_set1_for_predicitions.csv'
    mutation_csv = None
    
    if len(sys.argv) > 1:
        ligand_csv = sys.argv[1]
    if len(sys.argv) > 2:
        mutation_csv = sys.argv[2]
    
    # Run predictions
    try:
        results = predict_all_mutations(ligand_csv, mutation_csv)
        print("\n✅ PREDICTION COMPLETE!")
    except Exception as e:
        print(f"\n❌ ERROR during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
