import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, rdMolDescriptors
from rdkit import RDLogger
import sys
import os

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Import feature generation functions from dummy script
from dummy_physchem_5f2 import (
    generate_lig_inter_features,
    generate_lig_intra_features
)

def load_prediction_resources():
    """Load model, scalers, and mutant encoder"""
    print("\n" + "="*80)
    print("LOADING MODEL AND SCALERS")
    print("="*80)
    
    resources = {}
    
    # Load model
    if os.path.exists('feedforward_model.h5'):
        resources['model'] = load_model('feedforward_model.h5')
        print("✓ Loaded feedforward model")
    else:
        print("❌ ERROR: feedforward_model.h5 not found")
        return None
    
    # Load scalers
    if os.path.exists('feature_scalers.pkl'):
        with open('feature_scalers.pkl', 'rb') as f:
            resources['scalers'] = pickle.load(f)
        print("✓ Loaded feature scalers")
    else:
        print("❌ ERROR: feature_scalers.pkl not found")
        return None
    
    # Load Y scalers
    if os.path.exists('y_scalers.pkl'):
        with open('y_scalers.pkl', 'rb') as f:
            y_scalers = pickle.load(f)
            resources['y_scaler1'] = y_scalers['y_scaler1']  # Activity
            resources['y_scaler2'] = y_scalers['y_scaler2']  # Docking
        print("✓ Loaded y_scalers (Activity & Docking)")
    else:
        print("❌ ERROR: y_scalers.pkl not found")
        return None
    
    # Load mutant encoder
    if os.path.exists('mutant_encoder.pkl'):
        with open('mutant_encoder.pkl', 'rb') as f:
            resources['mutant_encoder'] = pickle.load(f)
        print("✓ Loaded mutant encoder")
    else:
        print("❌ ERROR: mutant_encoder.pkl not found")
        return None
    
    return resources


def get_ligand_features(ligand_smiles_series):
    """Extract ligand features from SMILES"""
    lig_inter_list = []
    lig_intra_list = []
    valid_indices = []
    
    for idx, smiles in enumerate(ligand_smiles_series):
        if pd.isna(smiles):
            continue
            
        lig_inter = generate_lig_inter_features(smiles)
        lig_intra = generate_lig_intra_features(smiles)
        
        if lig_inter is not None and lig_intra is not None:
            lig_inter_list.append(lig_inter)
            lig_intra_list.append(lig_intra)
            valid_indices.append(idx)
    
    if len(valid_indices) == 0:
        return None, None, []
    
    return np.array(lig_inter_list), np.array(lig_intra_list), valid_indices


def predict_for_mutation(ligand_names, ligand_smiles, mutation_name, resources):
    """Generate predictions for all ligands against a specific mutation"""
    
    print(f"\n{'='*80}")
    print(f"Processing Mutation: {mutation_name}")
    print(f"{'='*80}")
    
    model = resources['model']
    scalers = resources['scalers']
    y_scaler1 = resources['y_scaler1']
    y_scaler2 = resources['y_scaler2']
    mutant_encoder = resources['mutant_encoder']
    
    # Encode mutation
    if mutation_name not in mutant_encoder.classes_:
        print(f"  ⚠️ WARNING: {mutation_name} not in training data, skipping...")
        return None
    
    mutant_id_val = mutant_encoder.transform([mutation_name])[0]
    
    # Extract ligand features
    lig_inter, lig_intra, valid_idx = get_ligand_features(ligand_smiles)
    
    if len(valid_idx) == 0:
        print(f"  ❌ ERROR: No valid ligands for {mutation_name}")
        return None
    
    # Create mutant ID array
    mutant_ids = np.full((len(valid_idx),), mutant_id_val)
    
    # Scale features
    scaled_lig_inter = scalers['lig_inter'].transform(lig_inter)
    scaled_lig_intra = scalers['lig_intra'].transform(lig_intra)
    
    # Predict
    predictions = model.predict([
        mutant_ids,
        scaled_lig_inter,
        scaled_lig_intra
    ], verbose=0)
    
    pred_activity = predictions[0].flatten()
    pred_docking = predictions[1].flatten()
    
    # Inverse transform
    pred_activity = y_scaler1.inverse_transform(pred_activity.reshape(-1, 1)).flatten()
    pred_activity = np.expm1(pred_activity)  # Undo log1p
    
    pred_docking = y_scaler2.inverse_transform(pred_docking.reshape(-1, 1)).flatten()
    
    print(f"  ✓ Generated {len(pred_activity)} predictions")
    print(f"    Mean Activity: {pred_activity.mean():.2f}")
    print(f"    Mean Docking: {pred_docking.mean():.2f}")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'mutation_name': [mutation_name] * len(pred_activity),
        'ligand_name': [ligand_names[i] for i in valid_idx],
        'ligand_smiles': [ligand_smiles.iloc[i] for i in valid_idx],
        'predicted_activity': pred_activity,
        'predicted_docking': pred_docking
    })
    
    return results_df


def predict_all_mutations(ligand_csv_path, mutation_profiles_csv_path=None):
    """Main prediction function"""
    
    print("\n" + "="*80)
    print("PREDICTION SCRIPT FOR DUMMY PHYSCHEM MODEL")
    print("="*80)
    
    # Load resources
    resources = load_prediction_resources()
    if resources is None:
        print("❌ CRITICAL ERROR: Failed to load models/scalers. Exiting.")
        sys.exit(1)
    
    # Load ligands
    print(f"\nLoading ligands from: {ligand_csv_path}")
    df_ligands = pd.read_csv(ligand_csv_path)
    
    # Get ligand names and SMILES
    ligand_names = df_ligands.iloc[:, 0].values
    
    if 'smiles' in df_ligands.columns:
        ligand_smiles = df_ligands['smiles']
    elif df_ligands.shape[1] >= 2:
        print("  Note: 'smiles' column not found, using 2nd column as SMILES.")
        ligand_smiles = df_ligands.iloc[:, 1]
    else:
        print("❌ ERROR: CSV must have 'smiles' column or at least 2 columns!")
        sys.exit(1)
    
    print(f"✓ Loaded {len(ligand_smiles)} ligands")
    
    # Load mutation profiles
    if mutation_profiles_csv_path is None:
        print("\nSearching for training data in current directory...")
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        
        # Try common names
        possible_names = ['df_3_shuffled.csv', 'training_data.csv', 'df_train.csv']
        
        df_train = None
        for filename in possible_names:
            if filename in csv_files:
                try:
                    df_train = pd.read_csv(filename)
                    print(f"✓ Loaded training data from: {filename}")
                    break
                except:
                    continue
        
        if df_train is None:
            print("❌ ERROR: Could not find training data file")
            sys.exit(1)
    else:
        df_train = pd.read_csv(mutation_profiles_csv_path)
    
    df_train.columns = df_train.columns.str.strip()
    
    # Get unique mutations
    if 'tkd' not in df_train.columns:
        print("❌ ERROR: 'tkd' column not found in training data")
        sys.exit(1)
    
    unique_mutations = df_train['tkd'].dropna().unique()
    print(f"✓ Found {len(unique_mutations)} unique mutations")
    
    # Process each mutation
    all_results = []
    
    for mutation_name in unique_mutations:
        results_df = predict_for_mutation(
            ligand_names,
            ligand_smiles,
            mutation_name,
            resources
        )
        
        if results_df is not None:
            all_results.append(results_df)
    
    if len(all_results) == 0:
        print("\n❌ ERROR: No predictions were generated!")
        sys.exit(1)
    
    # Combine and save
    combined_results = pd.concat(all_results, ignore_index=True)
    output_file = 'predictions_dummy_physchem.csv'
    combined_results.to_csv(output_file, index=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"  Total predictions: {len(combined_results)}")
    print(f"  Mutations: {combined_results['mutation_name'].nunique()}")
    print(f"  Ligands: {combined_results['ligand_name'].nunique()}")
    
    return combined_results


if __name__ == "__main__":
    # Default paths - modify as needed
    ligand_csv = '/mnt/d/Publications/project_insilico/activity_physicochem_descriptor/placebo_blind_set1_for_predicitions.csv'
    mutation_csv = None
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        ligand_csv = sys.argv[1]
    if len(sys.argv) > 2:
        mutation_csv = sys.argv[2]
    
    predict_all_mutations(ligand_csv, mutation_csv)