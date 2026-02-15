#!/usr/bin/env python3
"""
Prediction script for adv_physchem_chemerta_crossattention2.py
ChemBERTa with Cross-Attention

This script loads trained hierarchical models with ChemBERTa embeddings and cross-attention
and makes predictions on new SMILES data.

Requirements:
    pip install transformers torch
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TensorFlow/Keras imports
from tensorflow.keras.models import load_model, Model


from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from numpy.linalg import norm

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value for zero denominator"""
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    else:
        result = np.where(denominator != 0, numerator / denominator, default)
        return result

def generate_lig_inter_features(smiles): #Intermolecular Ligand, input smiles ligand, returns np array
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    
    try:
        #Hydrogen Bonding
        features.append(Lipinski.NumHDonors(mol))
        features.append(Lipinski.NumHAcceptors(mol))
        features.append(Lipinski.NHOHCount(mol))
        features.append(Lipinski.NOCount(mol))
        features.append(rdMolDescriptors.CalcNumHBD(mol))
        features.append(rdMolDescriptors.CalcNumHBA(mol))
        
        #Electrostatic bonding
        features.append(Descriptors.MaxPartialCharge(mol))
        features.append(Descriptors.MinPartialCharge(mol))
        features.append(Descriptors.MaxAbsPartialCharge(mol))
        features.append(Descriptors.MaxPartialCharge(mol) - Descriptors.MinPartialCharge(mol))
        features.append(Descriptors.MinAbsPartialCharge(mol))
        
        #Polar surface
        features.append(MolSurf.TPSA(mol))
        
        features.append(MolSurf.LabuteASA(mol))

        features.append(Crippen.MolMR(mol))
        
        #Size & Rigidity
        features.append(Descriptors.MolWt(mol))
        features.append(Lipinski.HeavyAtomCount(mol))
        features.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        
        features.append(Crippen.MolLogP(mol))
        features.append(Descriptors.FractionCSP3(mol))
        features.append(Lipinski.NumAromaticRings(mol))
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        features.append(aromatic_atoms)
        
        # Pi-Pi stacking 
        features.append(Descriptors.NumAromaticCarbocycles(mol))
        features.append(Descriptors.NumAromaticHeterocycles(mol))

        #Halogen
        features.append(Fragments.fr_halogen(mol))

        #Flexibility
        features.append(Lipinski.NumRotatableBonds(mol))

        return np.array(features)
        
    except Exception as e:
        print(f"Error in lig_inter: {str(e)}")
        return None

def generate_mut_inter_features(smiles):
    return generate_lig_inter_features(smiles)

def generate_lig_intra_features(smiles): #Intramolecular Ligand
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    
    try:
        #Covalent bond
        num_bonds = mol.GetNumBonds()
        features.append(num_bonds)
        
        single_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 1.0)
        double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 2.0)
        triple_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 3.0)
        aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        features.extend([single_bonds, double_bonds, triple_bonds, aromatic_bonds])
        
        avg_bond_order = np.mean([bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]) if num_bonds > 0 else 0
        features.append(avg_bond_order)
        
        #Rigidity
        features.append(Lipinski.NumRotatableBonds(mol))
        features.append(Lipinski.RingCount(mol))
        features.append(Lipinski.NumAromaticRings(mol))
        
        rigid_bonds = sum(1 for bond in mol.GetBonds() if bond.IsInRing())
        fraction_rigid = rigid_bonds / num_bonds if num_bonds > 0 else 0
        features.append(fraction_rigid)
        
        # Pi-Pi bonding 
        features.append(Descriptors.NumAromaticCarbocycles(mol))
        features.append(Descriptors.NumAromaticHeterocycles(mol))
        
        #Hybridization
        sp2_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2)
        sp3_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3)
        sp_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP)
        features.extend([sp_carbons, sp2_carbons, sp3_carbons])
        
        #Ring strain
        ring_sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
        avg_ring_size = np.mean(ring_sizes) if ring_sizes else 0
        min_ring_size = min(ring_sizes) if ring_sizes else 0
        features.extend([avg_ring_size, min_ring_size])
        
        three_member_rings = sum(1 for size in ring_sizes if size == 3)
        four_member_rings = sum(1 for size in ring_sizes if size == 4)
        features.extend([three_member_rings, four_member_rings])
        
        #Complexity
        features.append(GraphDescriptors.BertzCT(mol))
        features.append(GraphDescriptors.Kappa1(mol))
        features.append(GraphDescriptors.Kappa2(mol))
        features.append(GraphDescriptors.Kappa3(mol))
        features.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
        features.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error in lig_intra: {str(e)}")
        return None

def generate_mut_intra_features(smiles):
    return generate_lig_intra_features(smiles)


def calculate_similarity_metrics(vec1, vec2):
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    norm1 = norm(vec1)
    norm2 = norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return {'cosine_similarity': 0.0, 'sine_dissimilarity': 0.0, 'dot_product': 0.0}
    cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    sine_of_angle = np.sqrt(max(0.0, 1 - cosine_sim**2))
    return {'cosine_similarity': float(cosine_sim), 'sine_dissimilarity': float(sine_of_angle), 'dot_product': float(np.dot(vec1, vec2))}


def calculate_fp_metrics(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return {'dice_sim': 0.0, 'tanimato': 0.0}
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    dice_sim = DataStructs.DiceSimilarity(fp1, fp2)
    tanimato = DataStructs.TanimotoSimilarity(fp1, fp2)
    return {'dice_sim': dice_sim, 'tanimato': tanimato}

def generate_inter_interaction_features(lig_inter, mut_inter):
    features = []
    metrics = calculate_similarity_metrics(lig_inter, mut_inter)
    features.append(metrics['cosine_similarity'])
    features.append(metrics['sine_dissimilarity'])
    return np.array(features)

def generate_intra_interaction_features(lig_intra, mut_intra):
    features = []
    metrics = calculate_similarity_metrics(lig_intra, mut_intra)
    features.append(metrics['cosine_similarity'])
    features.append(metrics['sine_dissimilarity'])
    return np.array(features)

def generate_final_interaction_features(lig_smiles, mut_smiles):
    features = []
    fp_inter_metrics = calculate_fp_metrics(lig_smiles, mut_smiles)
    features.extend([fp_inter_metrics['dice_sim'], fp_inter_metrics['tanimato']])
    return np.array(features)

def generate_custom_features(lig_inter, mut_inter, lig_intra, mut_intra):
    # Same implementation as original
    lig_mut_inter = []
    lig_mut_intra = []
    lig_mut_mix_inter_intra = []
    
    H_linear_lipinski = safe_divide(lig_inter[0] * mut_inter[1], mut_inter[0], default=0.0)
    lig_mut_inter.append(H_linear_lipinski)
    H_linear_total = safe_divide(lig_inter[4] * mut_inter[5], mut_inter[4], default=0.0)
    lig_mut_inter.append(H_linear_total)
    
    H_path = safe_divide(safe_divide(lig_inter[0] * mut_inter[1], mut_inter[0], default=0.0), mut_intra[21], default=0.0)
    lig_mut_mix_inter_intra.append(H_path)
    
    H_strength = safe_divide(lig_inter[0] * mut_inter[1], lig_inter[1], default=0.0) + safe_divide(lig_inter[1] * mut_inter[0], mut_inter[1], default=0.0)
    lig_mut_inter.append(H_strength)
    
    H_strength_total = safe_divide(lig_inter[4] * mut_inter[5], lig_inter[4], default=0.0) + safe_divide(lig_inter[5] * mut_inter[4], mut_inter[5], default=0.0)
    lig_mut_inter.append(H_strength_total)
    
    H_frac_lipinski = safe_divide(lig_inter[0], lig_inter[1], default=0.0) + safe_divide(mut_inter[1], mut_inter[0], default=0.0)
    lig_mut_inter.append(H_frac_lipinski)
    
    H_frac_total = safe_divide(lig_inter[4], lig_inter[5], default=0.0) + safe_divide(mut_inter[5], mut_inter[4], default=0.0)
    lig_mut_inter.append(H_frac_total)
    
    c_linear1_size1 = safe_divide(lig_inter[6], lig_intra[14], default=0.0) * safe_divide(mut_inter[7], mut_intra[14], default=0.0)
    lig_mut_mix_inter_intra.append(c_linear1_size1)
    
    c_linear2_size1 = safe_divide(lig_inter[7], lig_intra[14], default=0.0) * safe_divide(mut_inter[6], mut_intra[14], default=0.0)
    lig_mut_mix_inter_intra.append(c_linear2_size1)
    
    c_total = (c_linear1_size1 ** 2) + (c_linear2_size1 ** 2)
    lig_mut_mix_inter_intra.append(c_total)
    
    c_diff = ((lig_inter[6]) - (mut_inter[7])) - ((mut_inter[6]) - (lig_inter[7]))
    lig_mut_inter.append(c_diff)
    
    c_tpsa_diff = lig_inter[11] - mut_inter[11]
    lig_mut_inter.append(c_tpsa_diff)
    
    c_crip_logh = lig_inter[17] - mut_inter[17]
    lig_mut_inter.append(c_crip_logh)
    
    frac_tpsa_logH = safe_divide(lig_inter[11] * mut_inter[11], lig_inter[17] * mut_inter[17], default=0.0)
    lig_mut_inter.append(frac_tpsa_logH)
    
    pi_pi_ratio1 = safe_divide(lig_inter[21] + lig_inter[22] + mut_inter[21] + mut_inter[22], lig_intra[15] + mut_intra[15], default=0.0)
    lig_mut_mix_inter_intra.append(pi_pi_ratio1)
    
    pi_pi_ratio2 = safe_divide(lig_inter[21] + lig_inter[22] + mut_inter[21] + mut_inter[22], lig_intra[22] + mut_intra[22], default=0.0)
    lig_mut_mix_inter_intra.append(pi_pi_ratio2)
    
    bond_rigid = safe_divide(lig_intra[2] + lig_intra[3] + lig_intra[4], lig_intra[0], default=0.0) + safe_divide(mut_intra[2] + mut_intra[3] + mut_intra[4], mut_intra[0], default=0.0)
    bond_single = safe_divide(lig_intra[1], lig_intra[0], default=0.0) + safe_divide(mut_intra[1], mut_intra[0], default=0.0)
    bond_diff = (bond_single - bond_rigid) ** 2
    lig_mut_intra.append(bond_diff)
    
    hybridisation_lig = safe_divide(lig_intra[12] + lig_intra[13], lig_intra[14] + lig_intra[12] + lig_intra[13], default=0.0)
    hybridisation_mut = safe_divide(mut_intra[12] + mut_intra[13], mut_intra[14] + mut_intra[12] + mut_intra[13], default=0.0)
    hybridisation_diff = (hybridisation_mut - hybridisation_lig) ** 2
    lig_mut_intra.append(hybridisation_diff)
    
    bertz_ratio = safe_divide(lig_intra[21], mut_intra[21], default=0.0)
    lig_mut_intra.append(bertz_ratio)
    
    return lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra

def generate_hierarchical_features(ligand_smiles_series, mutation_smiles_series):
    print('\nGenerating hierarchical features...')
    ligand_cache = {}
    mutation_cache = {}
    interaction_cache = {}

    lig_inter_list = []
    mut_inter_list = []
    inter_interaction_list = []
    lig_intra_list = []
    mut_intra_list = []
    intra_interaction_list = []
    lig_mut_mix_inter_intra_list = []
    fp_interaction_list = []
    valid_indices = []

    for idx, (lig_smi, mut_smi) in enumerate(zip(ligand_smiles_series, mutation_smiles_series)):
        if idx % 50 == 0:
            print(f'  Processing sample {idx}/{len(ligand_smiles_series)}...')
        if lig_smi in ligand_cache:
            lig_inter, lig_intra = ligand_cache[lig_smi]
        else:
            lig_inter = generate_lig_inter_features(lig_smi)
            lig_intra = generate_lig_intra_features(lig_smi)
            ligand_cache[lig_smi] = (lig_inter, lig_intra)

        if mut_smi in mutation_cache:
            mut_inter, mut_intra = mutation_cache[mut_smi]
        else:
            mut_inter = generate_mut_inter_features(mut_smi)
            mut_intra = generate_mut_intra_features(mut_smi)
            mutation_cache[mut_smi] = (mut_inter, mut_intra)

        if any(f is None for f in [lig_inter, mut_inter, lig_intra, mut_intra]):
            continue

        pair_key = (lig_smi, mut_smi)
        if pair_key in interaction_cache:
            lig_mut_mix_inter_intra, inter_interaction, intra_interaction, final_fp_interaction = interaction_cache[pair_key]
        else:
            lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra = generate_custom_features(lig_inter, mut_inter, lig_intra, mut_intra)
            inter_interaction = generate_inter_interaction_features(lig_inter, mut_inter)
            intra_interaction = generate_intra_interaction_features(lig_intra, mut_intra)
            if len(lig_mut_inter) > 0:
                inter_interaction = np.concatenate([np.array(lig_mut_inter), inter_interaction])
            if len(lig_mut_intra) > 0:
                intra_interaction = np.concatenate([np.array(lig_mut_intra), intra_interaction])
            final_fp_interaction = generate_final_interaction_features(lig_smi, mut_smi)
            interaction_cache[pair_key] = (lig_mut_mix_inter_intra, inter_interaction, intra_interaction, final_fp_interaction)

        lig_inter_list.append(lig_inter)
        mut_inter_list.append(mut_inter)
        inter_interaction_list.append(inter_interaction)
        lig_intra_list.append(lig_intra)
        mut_intra_list.append(mut_intra)
        intra_interaction_list.append(intra_interaction)
        fp_interaction_list.append(final_fp_interaction)
        lig_mut_mix_inter_intra_list.append(np.array(lig_mut_mix_inter_intra))
        valid_indices.append(idx)

    print(f'  Successfully generated features for {len(valid_indices)} samples')
    return {
        'lig_inter': np.array(lig_inter_list),
        'mut_inter': np.array(mut_inter_list),
        'inter_interaction': np.array(inter_interaction_list),
        'lig_intra': np.array(lig_intra_list),
        'mut_intra': np.array(mut_intra_list),
        'intra_interaction': np.array(intra_interaction_list),
        'lig_mut_mix_inter_intra': np.array(lig_mut_mix_inter_intra_list),
        'final_fp_interaction': np.array(fp_interaction_list),
        'valid_indices': valid_indices
    }


print("=" * 80)
print("PREDICTION SCRIPT FOR CHEMBERTA CROSS-ATTENTION MODEL")
print("Hierarchical 6-site RNN with ChemBERTa embeddings and cross-attention")
print("=" * 80)

# ============================================================================
# CHEMBERTA EMBEDDING FUNCTIONS
# ============================================================================

def get_device():
    """Get available device (GPU if available, else CPU)"""
    try:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        return 'cpu'

def load_chemberta(model_name='seyonec/ChemBERTa-zinc-base-v1', device=None):
    """
    Load ChemBERTa model and tokenizer
    
    Args:
        model_name: HuggingFace model name
        device: torch device (auto-detected if None)
    
    Returns:
        tokenizer, model, device
    """
    print(f"\nLoading ChemBERTa model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    if device is None:
        device = get_device()
    
    model.to(device)
    model.eval()
    
    print(f"  ✓ ChemBERTa loaded on {device}")
    return tokenizer, model, device

def get_chemberta_embeddings(smiles_list, tokenizer, model, device, batch_size=32):
    """
    Generate ChemBERTa embeddings for a list of SMILES strings
    
    Args:
        smiles_list: List of SMILES strings
        tokenizer: ChemBERTa tokenizer
        model: ChemBERTa model
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        numpy array of shape (n_smiles, 768)
    """
    # Handle single string input
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    
    # Tokenize all SMILES
    inputs = tokenizer(smiles_list, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    # Generate embeddings
    embeddings_list = []
    with torch.no_grad():
        for batch in loader:
            ids = batch[0].to(device)
            mask = batch[1].to(device)
            
            # Forward pass
            outputs = model(input_ids=ids, attention_mask=mask, return_dict=True)
            
            # Get pooled output (CLS token embedding)
            pooled = getattr(outputs, 'pooler_output', None)
            
            # If no pooler_output, use mean pooling of last hidden state
            if pooled is None:
                last_hidden = outputs.last_hidden_state
                mask_expanded = mask.unsqueeze(-1).float()
                summed = (last_hidden * mask_expanded).sum(1)
                lengths = mask_expanded.sum(1).clamp(min=1e-9)
                pooled = summed / lengths
            
            embeddings_list.append(pooled.cpu())
    
    # Concatenate all embeddings
    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings.numpy()

# ============================================================================
# CONFIGURATION
# ============================================================================

MUTATION_SITES = ['Full', 'ATP_POCKET', 'P_LOOP', 'C_HELIX', 'DFG_A_LOOP', 'HRD_CAT']
SITE_COLUMNS = [
    'smiles_full_egfr', 'smiles 718_862_atp_pocket', 'smiles_p_loop',
    'smiles_c_helix', 'smiles_l858r_a_loop_dfg_motif', 'smiles_catalytic_hrd_motif'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_models_and_scalers(model_dir):
    """Load all required models and scalers"""
    
    print("\nLoading models and scalers...")
    
    # Load hierarchical models for each site
    hierarchical_models = {}
    for site_name in MUTATION_SITES:
        model_path = os.path.join(model_dir, f'hierarchical_model_{site_name}.h5')
        if os.path.exists(model_path):
            hierarchical_models[site_name] = load_model(model_path, compile=False)
            print(f"  ✓ Loaded {site_name} model")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load RNN model (may be named differently)
    rnn_path = os.path.join(model_dir, 'rnn_model_keras.h5')
    if not os.path.exists(rnn_path):
        # Try alternative name
        rnn_path = os.path.join(model_dir, 'rnn_sequential_model.h5')
    
    if os.path.exists(rnn_path):
        rnn_model = load_model(rnn_path, compile=False)
        print(f"  ✓ Loaded RNN model")
    else:
        raise FileNotFoundError(f"RNN model not found")
    
    # Load feature scalers
    scalers_path = os.path.join(model_dir, 'feature_scalers.pkl')
    if not os.path.exists(scalers_path):
        # Try alternative name
        scalers_path = os.path.join(model_dir, 'all_scalers.pkl')
    
    with open(scalers_path, 'rb') as f:
        all_scalers = pickle.load(f)
    print(f"  ✓ Loaded feature scalers")
    
    # Load y scalers
    y1_path = os.path.join(model_dir, 'y_scaler1.pkl')
    y2_path = os.path.join(model_dir, 'y_scaler2.pkl')
    
    if os.path.exists(y1_path) and os.path.exists(y2_path):
        with open(y1_path, 'rb') as f:
            y_scaler1 = pickle.load(f)
        with open(y2_path, 'rb') as f:
            y_scaler2 = pickle.load(f)
    else:
        # Try combined file
        with open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb') as f:
            y_scalers = pickle.load(f)
        y_scaler1 = y_scalers['y_scaler1']
        y_scaler2 = y_scalers['y_scaler2']
    
    print(f"  ✓ Loaded y scalers")
    
    # Load ChemBERTa embedding scaler
    chem_scaler_path = os.path.join(model_dir, 'chem_emb_scaler.pkl')
    if not os.path.exists(chem_scaler_path):
        chem_scaler_path = os.path.join(model_dir, 'chembert_scalers.pkl')
    
    with open(chem_scaler_path, 'rb') as f:
        chem_emb_scaler = pickle.load(f)
    print(f"  ✓ Loaded ChemBERTa embedding scaler")
    
    return hierarchical_models, rnn_model, all_scalers, y_scaler1, y_scaler2, chem_emb_scaler

# ============================================================================
# UPDATED EVALUATION AND PLOTTING FUNCTION
# ============================================================================

def evaluate_and_plot(df_results, output_dir, model_name):
    """
    Evaluate predictions and generate comprehensive plots with statistics
    
    Updates:
    - Save metrics to CSV table
    - Create individual plots for each mutation
    - Generate Pearson correlation plots per mutation
    - Save all plots separately
    """
    
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    
    # Create metrics directory
    metrics_dir = os.path.join(output_dir, 'metrics')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # ========================================================================
    # 1. OVERALL METRICS
    # ========================================================================
    mae_act = mean_absolute_error(df_results['actual_activity'], df_results['predicted_activity'])
    rmse_act = np.sqrt(mean_squared_error(df_results['actual_activity'], df_results['predicted_activity']))
    mae_dock = mean_absolute_error(df_results['actual_docking'], df_results['predicted_docking'])
    rmse_dock = np.sqrt(mean_squared_error(df_results['actual_docking'], df_results['predicted_docking']))
    
    # Calculate Pearson correlation for overall
    pearson_act, pval_act = pearsonr(df_results['actual_activity'], df_results['predicted_activity'])
    pearson_dock, pval_dock = pearsonr(df_results['actual_docking'], df_results['predicted_docking'])
    
    print(f"\nOverall Performance:")
    print(f"  Activity  - MAE: {mae_act:.4f}, RMSE: {rmse_act:.4f}, Pearson R: {pearson_act:.4f} (p={pval_act:.4e})")
    print(f"  Docking   - MAE: {mae_dock:.4f}, RMSE: {rmse_dock:.4f}, Pearson R: {pearson_dock:.4f} (p={pval_dock:.4e})")
    
    # ========================================================================
    # 2. PER-MUTATION METRICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("PER-MUTATION METRICS")
    print("=" * 80)
    
    # Store metrics for CSV
    metrics_data = []
    
    # Add overall metrics first
    metrics_data.append({
        'Mutation': 'Overall',
        'N_Samples': len(df_results),
        'Activity_MAE': mae_act,
        'Activity_RMSE': rmse_act,
        'Activity_Pearson_R': pearson_act,
        'Activity_Pearson_pval': pval_act,
        'Docking_MAE': mae_dock,
        'Docking_RMSE': rmse_dock,
        'Docking_Pearson_R': pearson_dock,
        'Docking_Pearson_pval': pval_dock
    })
    
    # Process each mutation
    mutations = sorted(df_results['tkd'].unique())
    
    for mutation in mutations:
        mut_data = df_results[df_results['tkd'] == mutation]
        n_samples = len(mut_data)
        
        if n_samples < 2:
            print(f"\n{mutation}: Insufficient data (n={n_samples}), skipping")
            continue
        
        # Calculate metrics
        mae_a = mean_absolute_error(mut_data['actual_activity'], mut_data['predicted_activity'])
        rmse_a = np.sqrt(mean_squared_error(mut_data['actual_activity'], mut_data['predicted_activity']))
        mae_d = mean_absolute_error(mut_data['actual_docking'], mut_data['predicted_docking'])
        rmse_d = np.sqrt(mean_squared_error(mut_data['actual_docking'], mut_data['predicted_docking']))
        
        # Pearson correlations
        try:
            pearson_a, pval_a = pearsonr(mut_data['actual_activity'], mut_data['predicted_activity'])
            pearson_d, pval_d = pearsonr(mut_data['actual_docking'], mut_data['predicted_docking'])
        except:
            pearson_a, pval_a = np.nan, np.nan
            pearson_d, pval_d = np.nan, np.nan
        
        print(f"\n{mutation} (n={n_samples}):")
        print(f"  Activity  - MAE: {mae_a:.4f}, RMSE: {rmse_a:.4f}, Pearson R: {pearson_a:.4f}")
        print(f"  Docking   - MAE: {mae_d:.4f}, RMSE: {rmse_d:.4f}, Pearson R: {pearson_d:.4f}")
        
        # Store in metrics list
        metrics_data.append({
            'Mutation': mutation,
            'N_Samples': n_samples,
            'Activity_MAE': mae_a,
            'Activity_RMSE': rmse_a,
            'Activity_Pearson_R': pearson_a,
            'Activity_Pearson_pval': pval_a,
            'Docking_MAE': mae_d,
            'Docking_RMSE': rmse_d,
            'Docking_Pearson_R': pearson_d,
            'Docking_Pearson_pval': pval_d
        })
    
    # ========================================================================
    # 3. SAVE METRICS TO CSV
    # ========================================================================
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(metrics_dir, f'{model_name}_metrics_summary.csv')
    metrics_df.to_csv(metrics_csv_path, index=False, float_format='%.6f')
    print(f"\n✓ Metrics saved to: {metrics_csv_path}")
    
    # ========================================================================
    # 4. CREATE OVERALL COMBINED PLOT (2x2 layout)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Activity scatter plot
    axes[0, 0].scatter(df_results['actual_activity'], df_results['predicted_activity'], 
                      alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    axes[0, 0].plot([df_results['actual_activity'].min(), df_results['actual_activity'].max()],
                   [df_results['actual_activity'].min(), df_results['actual_activity'].max()],
                   'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual Activity', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Activity', fontsize=11)
    axes[0, 0].set_title(f'Activity Prediction (Overall)\nMAE={mae_act:.4f}, RMSE={rmse_act:.4f}, R={pearson_act:.4f}', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Docking scatter plot
    axes[0, 1].scatter(df_results['actual_docking'], df_results['predicted_docking'], 
                      alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    axes[0, 1].plot([df_results['actual_docking'].min(), df_results['actual_docking'].max()],
                   [df_results['actual_docking'].min(), df_results['actual_docking'].max()],
                   'r--', lw=2, label='Perfect prediction')
    axes[0, 1].set_xlabel('Actual Docking Score', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Docking Score', fontsize=11)
    axes[0, 1].set_title(f'Docking Prediction (Overall)\nMAE={mae_dock:.4f}, RMSE={rmse_dock:.4f}, R={pearson_dock:.4f}', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Activity residuals by mutation
    mutation_colors = plt.cm.tab10(np.linspace(0, 1, len(mutations)))
    for idx, mutation in enumerate(mutations):
        mut_data = df_results[df_results['tkd'] == mutation]
        residuals = mut_data['actual_activity'] - mut_data['predicted_activity']
        axes[1, 0].scatter(mut_data['predicted_activity'], residuals, 
                         label=mutation, alpha=0.6, s=20, c=[mutation_colors[idx]])
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Activity', fontsize=11)
    axes[1, 0].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    axes[1, 0].set_title('Activity Residuals by Mutation', fontsize=12)
    axes[1, 0].legend(fontsize='small', loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Docking residuals by mutation
    for idx, mutation in enumerate(mutations):
        mut_data = df_results[df_results['tkd'] == mutation]
        residuals = mut_data['actual_docking'] - mut_data['predicted_docking']
        axes[1, 1].scatter(mut_data['predicted_docking'], residuals, 
                         label=mutation, alpha=0.6, s=20, c=[mutation_colors[idx]])
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Docking Score', fontsize=11)
    axes[1, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    axes[1, 1].set_title('Docking Residuals by Mutation', fontsize=12)
    axes[1, 1].legend(fontsize='small', loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    overall_plot_file = os.path.join(plots_dir, f'{model_name}_overall_combined.png')
    plt.savefig(overall_plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Overall combined plot saved to: {overall_plot_file}")
    plt.close()
    
    # ========================================================================
    # 5. CREATE INDIVIDUAL PLOTS FOR EACH MUTATION
    # ========================================================================
    print("\nGenerating individual mutation plots...")
    
    for mutation in mutations:
        mut_data = df_results[df_results['tkd'] == mutation]
        
        if len(mut_data) < 2:
            continue
        
        # Get metrics for this mutation
        mut_metrics = metrics_df[metrics_df['Mutation'] == mutation].iloc[0]
        
        # Create 2x2 subplot for each mutation
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Mutation: {mutation} (n={len(mut_data)})', fontsize=14, fontweight='bold')
        
        # Activity scatter
        axes[0, 0].scatter(mut_data['actual_activity'], mut_data['predicted_activity'], 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='steelblue')
        min_val = min(mut_data['actual_activity'].min(), mut_data['predicted_activity'].min())
        max_val = max(mut_data['actual_activity'].max(), mut_data['predicted_activity'].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        axes[0, 0].set_xlabel('Actual Activity', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Activity', fontsize=11)
        axes[0, 0].set_title(f'Activity Prediction\nMAE={mut_metrics["Activity_MAE"]:.4f}, RMSE={mut_metrics["Activity_RMSE"]:.4f}', fontsize=11)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Docking scatter
        axes[0, 1].scatter(mut_data['actual_docking'], mut_data['predicted_docking'], 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='darkorange')
        min_val = min(mut_data['actual_docking'].min(), mut_data['predicted_docking'].min())
        max_val = max(mut_data['actual_docking'].max(), mut_data['predicted_docking'].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        axes[0, 1].set_xlabel('Actual Docking Score', fontsize=11)
        axes[0, 1].set_ylabel('Predicted Docking Score', fontsize=11)
        axes[0, 1].set_title(f'Docking Prediction\nMAE={mut_metrics["Docking_MAE"]:.4f}, RMSE={mut_metrics["Docking_RMSE"]:.4f}', fontsize=11)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Activity residuals
        residuals_act = mut_data['actual_activity'] - mut_data['predicted_activity']
        axes[1, 0].scatter(mut_data['predicted_activity'], residuals_act, 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='steelblue')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Activity', fontsize=11)
        axes[1, 0].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
        axes[1, 0].set_title('Activity Residuals', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Docking residuals
        residuals_dock = mut_data['actual_docking'] - mut_data['predicted_docking']
        axes[1, 1].scatter(mut_data['predicted_docking'], residuals_dock, 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='darkorange')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Predicted Docking Score', fontsize=11)
        axes[1, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
        axes[1, 1].set_title('Docking Residuals', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save mutation-specific plot
        mutation_safe = mutation.replace('/', '_').replace('\\', '_')
        mutation_plot_file = os.path.join(plots_dir, f'{model_name}_mutation_{mutation_safe}.png')
        plt.savefig(mutation_plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {mutation_safe}.png")
        plt.close()
    
    # ========================================================================
    # 6. CREATE PEARSON CORRELATION PLOTS FOR EACH MUTATION
    # ========================================================================
    print("\nGenerating Pearson correlation plots...")
    
    pearson_dir = os.path.join(plots_dir, 'pearson_correlations')
    os.makedirs(pearson_dir, exist_ok=True)
    
    for mutation in mutations:
        mut_data = df_results[df_results['tkd'] == mutation]
        
        if len(mut_data) < 2:
            continue
        
        # Get metrics for this mutation
        mut_metrics = metrics_df[metrics_df['Mutation'] == mutation].iloc[0]
        
        # Create figure with 1x2 subplots for activity and docking
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Pearson Correlations - {mutation} (n={len(mut_data)})', fontsize=14, fontweight='bold')
        
        # Activity Pearson correlation
        axes[0].scatter(mut_data['actual_activity'], mut_data['predicted_activity'], 
                       alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='steelblue')
        
        # Add regression line
        z = np.polyfit(mut_data['actual_activity'], mut_data['predicted_activity'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(mut_data['actual_activity'].min(), mut_data['actual_activity'].max(), 100)
        axes[0].plot(x_line, p(x_line), "g-", linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
        
        # Perfect prediction line
        min_val = min(mut_data['actual_activity'].min(), mut_data['predicted_activity'].min())
        max_val = max(mut_data['actual_activity'].max(), mut_data['predicted_activity'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        axes[0].set_xlabel('Actual Activity', fontsize=12)
        axes[0].set_ylabel('Predicted Activity', fontsize=12)
        axes[0].set_title(f'Activity\nPearson R = {mut_metrics["Activity_Pearson_R"]:.4f}, p = {mut_metrics["Activity_Pearson_pval"]:.4e}', 
                         fontsize=11)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Docking Pearson correlation
        axes[1].scatter(mut_data['actual_docking'], mut_data['predicted_docking'], 
                       alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='darkorange')
        
        # Add regression line
        z = np.polyfit(mut_data['actual_docking'], mut_data['predicted_docking'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(mut_data['actual_docking'].min(), mut_data['actual_docking'].max(), 100)
        axes[1].plot(x_line, p(x_line), "g-", linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
        
        # Perfect prediction line
        min_val = min(mut_data['actual_docking'].min(), mut_data['predicted_docking'].min())
        max_val = max(mut_data['actual_docking'].max(), mut_data['predicted_docking'].max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        axes[1].set_xlabel('Actual Docking Score', fontsize=12)
        axes[1].set_ylabel('Predicted Docking Score', fontsize=12)
        axes[1].set_title(f'Docking\nPearson R = {mut_metrics["Docking_Pearson_R"]:.4f}, p = {mut_metrics["Docking_Pearson_pval"]:.4e}', 
                         fontsize=11)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save Pearson correlation plot
        mutation_safe = mutation.replace('/', '_').replace('\\', '_')
        pearson_plot_file = os.path.join(pearson_dir, f'{model_name}_pearson_{mutation_safe}.png')
        plt.savefig(pearson_plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: pearson_{mutation_safe}.png")
        plt.close()
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"✓ Metrics CSV: {metrics_csv_path}")
    print(f"✓ Overall plot: {overall_plot_file}")
    print(f"✓ Individual mutation plots: {plots_dir}")
    print(f"✓ Pearson correlation plots: {pearson_dir}")
    print("=" * 80)

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_predictions(input_csv, model_dir='.', output_dir='.'):
    """
    Make predictions using the trained dummy_physchem_5f2 model.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV file with 'smiles' and 'tkd' columns
    model_dir : str
        Directory containing model files (default: current directory)
    output_dir : str
        Directory to save prediction outputs (default: current directory)
        
    Returns:
    --------
    df_results : pd.DataFrame
        DataFrame with predictions
    """
    
    print(f"Loading prediction data from: {input_csv}")
    df_pred = pd.read_csv(input_csv)
    
    # Check required columns
    required_cols = ['smiles', 'tkd']
    if not all(col in df_pred.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain 'smiles' and 'tkd' columns. Found: {df_pred.columns.tolist()}")
    
    # Check if ground truth exists
    has_ground_truth = 'standard value' in df_pred.columns and 'dock' in df_pred.columns
    
    # Load Model Files
    print("\nLoading model files...")
    try:
        model = load_model(os.path.join(model_dir, 'feedforward_model.h5'))
        
        with open(os.path.join(model_dir, 'mutant_encoder.pkl'), 'rb') as f:
            mutant_encoder = pickle.load(f)
            
        with open(os.path.join(model_dir, 'mutant_mapping.pkl'), 'rb') as f:
            mutant_mapping = pickle.load(f)
            
        with open(os.path.join(model_dir, 'feature_scalers.pkl'), 'rb') as f:
            feature_scalers = pickle.load(f)
        
        with open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb') as f:
            y_scalers = pickle.load(f)
            
        y_scaler1 = y_scalers['y_scaler1']
        y_scaler2 = y_scalers['y_scaler2']
        
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please ensure 'feedforward_model.h5', 'mutant_encoder.pkl', 'mutant_mapping.pkl', 'feature_scalers.pkl', and 'y_scalers.pkl' are in the model directory.")
        return None

    print(f"Total prediction samples: {len(df_pred)}")
    
    # Prepare batch data
    lig_inter_list = []
    lig_intra_list = []
    mutant_id_list = []
    valid_indices = []
    
    print("Generating features...")
    for idx, row in df_pred.iterrows():
        lig_smiles = row['smiles']
        mutant_name = row['tkd']
        
        # Validate SMILES
        if pd.isna(lig_smiles) or lig_smiles == '':
            print(f"Warning: Empty SMILES at row {idx}, skipping")
            continue

        # Validate Mutation
        if pd.isna(mutant_name):
            print(f"Warning: Empty mutation at row {idx}, skipping")
            continue
            
        # 1. Generate Lib Inputs
        lig_inter = generate_lig_inter_features(lig_smiles)
        lig_intra = generate_lig_intra_features(lig_smiles)
        
        if lig_inter is None or lig_intra is None:
            print(f"Warning: Could not generate features for row {idx} (SMILES: {lig_smiles}), skipping")
            continue
            
        # 2. Encode Mutation
        # Training script uses simple integer encoding relative to sorted unique values in training
        if mutant_name in mutant_mapping:
            mut_id = mutant_mapping[mutant_name]
        else:
            print(f"Warning: Mutation '{mutant_name}' not seen in training. Using default ID 0.")
            mut_id = 0 # Default fallback
            
        lig_inter_list.append(lig_inter)
        lig_intra_list.append(lig_intra)
        mutant_id_list.append(mut_id)
        valid_indices.append(idx)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df_pred)} samples")
            
    if not valid_indices:
        print("Error: No valid samples found to predict.")
        return None
        
    # Convert to numpy arrays
    X_lig_inter = np.array(lig_inter_list)
    X_lig_intra = np.array(lig_intra_list)
    X_mutant = np.array(mutant_id_list)
    
    # Scale features
    # Note: Training script scales lig_inter and lig_intra. Mutant ID is used as is (Embedding layer handles it).
    X_lig_inter_scaled = feature_scalers['lig_inter'].transform(X_lig_inter)
    X_lig_intra_scaled = feature_scalers['lig_intra'].transform(X_lig_intra)
    
    print(f"Predicting for {len(valid_indices)} samples...")
    
    # Make Prediction
    # Model inputs: [mutant_input, inter_input, intra_input]
    predictions = model.predict(
        [X_mutant, X_lig_inter_scaled, X_lig_intra_scaled],
        verbose=1
    )
    
    # Raw outputs
    pred_activity_scaled = predictions[0].flatten()
    pred_docking_scaled = predictions[1].flatten()
    
    # Inverse Transform
    # Activity: log1p -> scaler -> model -> scaler inverse -> expm1
    pred_activity_log1p = y_scaler1.inverse_transform(pred_activity_scaled.reshape(-1, 1)).flatten()
    pred_activity = np.expm1(pred_activity_log1p)
    
    # Docking: scaler -> model -> scaler inverse
    pred_docking = y_scaler2.inverse_transform(pred_docking_scaled.reshape(-1, 1)).flatten()
    
    # Store Results
    results = []
    for i, original_idx in enumerate(valid_indices):
        row = df_pred.iloc[original_idx]
        res = {
            'smiles': row['smiles'],
            'tkd': row['tkd'],
            'predicted_activity': pred_activity[i],
            'predicted_docking': pred_docking[i]
        }
        
        if has_ground_truth:
            res['actual_activity'] = row['standard value']
            res['actual_docking'] = row['dock']
            
        # Keep other columns
        for col in df_pred.columns:
            if col not in res and col not in ['smiles', 'tkd', 'standard value', 'dock']:
                 res[col] = row[col]
                 
        results.append(res)
        
    df_results = pd.DataFrame(results)
    
    # Save output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'predictions_dummy_physchem_5f2.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    # Evaluation (if ground truth exists)
    if has_ground_truth and len(df_results) > 0:
        evaluate_and_plot(df_results, output_dir, 'dummy_physchem_5f2')
        
    return df_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predictions for ChemBERTa Cross-Attention model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--model_dir', type=str, default='.', help='Model directory')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    results = make_predictions(args.input, args.model_dir, args.output_dir)
    
    print(f"\n✓ Complete! Total predictions: {len(results)}")
    print(f"✓ Mutations covered: {results['mutation'].nunique()}")
