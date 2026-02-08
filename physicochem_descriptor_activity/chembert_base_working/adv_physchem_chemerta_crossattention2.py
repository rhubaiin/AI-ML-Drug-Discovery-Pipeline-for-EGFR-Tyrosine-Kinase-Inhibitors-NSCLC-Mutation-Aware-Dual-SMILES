#!/usr/bin/env python3
"""
adv_physchem_chemerta_crossattention2.py

Modified version of adv_physchem_chemberta_crossattention1.py.
FIX: Implements Cross-Attention using Keras MultiHeadAttention layer
to ensure the attention mechanism allows backpropagation updates during training.

Key Changes:
- Replaced PyTorch CrossAttentionLayer with Keras MultiHeadAttention.
- Integrated Attention directly into `priority_hierarchical_model`.
- ChemBERTa embeddings are passed as raw inputs and attended to dynamically.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from loguru import logger

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy.linalg import norm

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Concatenate, Multiply, 
    LSTM, GRU, Bidirectional, LeakyReLU, 
    MultiHeadAttention, LayerNormalization, Add, Reshape, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import torch
# We keep torch/transformers for initial ChemBERTa feature extraction
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(42)
tf.random.set_seed(42)


# Fixed safe_divide and feature generation functions
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


logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("adv_physchem_crossattention2_{time}.txt", rotation="500 MB", retention="10 days", compression="zip", level="DEBUG")

print("="*80)
print("RNN-LSTM INTEGRATED HIERARCHICAL MODEL WITH TRAINABLE CROSS-ATTENTION (KERAS)")
print("Strategy 3: Dynamic Cross-Attention Between Ligand and Each Mutation Site")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================
print("\nLoading datasets...")
# Using absolute paths as per original script
df_train = pd.read_csv(os.path.join(script_dir, 'df_3_shuffled.csv'))
df_control = pd.read_csv(os.path.join(script_dir, 'egfr_tki_valid_cleaned.csv'))
df_drugs = pd.read_csv(os.path.join(script_dir, 'drugs.csv'))

df_train.columns = df_train.columns.str.strip()

ligand_smiles = df_train['smiles']
full_smiles = df_train['smiles_full_egfr']
mutation_smiles = df_train['smiles 718_862_atp_pocket']
mut_hinge_p_loop = df_train['smiles_p_loop']
mut_helix = df_train['smiles_c_helix']
mut_dfg_a_loop = df_train['smiles_l858r_a_loop_dfg_motif']
mut_hrd_cat = df_train['smiles_catalytic_hrd_motif']
mutant = df_train['tkd']
activity_values = df_train['standard value']
docking_values = df_train['dock']

control_smiles = df_control['smiles_control']
control_name = df_control['id']
drug_smiles = df_drugs['smiles']

print(f"Training samples: {len(ligand_smiles)}")

valid_mask = ~(
    ligand_smiles.isna() | full_smiles.isna() | mutation_smiles.isna() | 
    mut_hinge_p_loop.isna() | mut_helix.isna() | mut_dfg_a_loop.isna() | 
    mut_hrd_cat.isna() | activity_values.isna() | docking_values.isna() 
)
valid_sample_count = valid_mask.sum()
print(f"\n✓ Valid samples: {valid_sample_count}/{len(df_train)}")
if valid_sample_count == 0:
    sys.exit(1)

df_train_valid = df_train[valid_mask].copy().reset_index(drop=True)
ligand_smiles_valid = df_train_valid['smiles']
full_smiles_valid = df_train_valid['smiles_full_egfr']
mutation_smiles_valid = df_train_valid['smiles 718_862_atp_pocket']
mut_hinge_p_loop_valid = df_train_valid['smiles_p_loop']
mut_helix_valid = df_train_valid['smiles_c_helix']
mut_dfg_a_loop_valid = df_train_valid['smiles_l858r_a_loop_dfg_motif']
mut_hrd_cat_valid = df_train_valid['smiles_catalytic_hrd_motif']
mutant_valid = df_train_valid['tkd']
activity_values_valid = df_train_valid['standard value']
activity_values2_valid = df_train_valid['dock'].values

mutation_profile_columns = [
    'smiles_full_egfr', 'smiles 718_862_atp_pocket', 'smiles_p_loop', 
    'smiles_c_helix', 'smiles_l858r_a_loop_dfg_motif', 'smiles_catalytic_hrd_motif', 'tkd'
]
unique_mutation_profiles = df_train_valid[mutation_profile_columns].drop_duplicates(subset=['tkd']).reset_index(drop=True)

# ============================================================================
# HELPER FUNCTIONS (Metrics & Features)
# ============================================================================

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

def get_device():
    try:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        return 'cpu'

def load_chemberta(model_name: str = 'seyonec/ChemBERTa-zinc-base-v1', device=None):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device

def get_chemberta_embeddings(smiles, tokenizer, model, device, batch_size=32):
    inputs = tokenizer(smiles, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset, batch_size=batch_size)
    embs = []
    with torch.no_grad():
        for batch in loader:
            ids = batch[0].to(device)
            mask = batch[1].to(device)
            out = model(input_ids=ids, attention_mask=mask, return_dict=True)
            pooled = getattr(out, 'pooler_output', None)
            if pooled is None:
                last = out.last_hidden_state
                maskf = mask.unsqueeze(-1).float()
                summed = (last * maskf).sum(1)
                lengths = maskf.sum(1).clamp(min=1e-9)
                pooled = summed / lengths
            embs.append(pooled.cpu())
    embs = torch.cat(embs, dim=0)
    return embs.numpy()

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

def build_priority_hierarchical_model(feature_dims):
    logger.info('=' * 80)
    logger.info('BUILDING PRIORITY HIERARCHICAL MODEL (with Keras MultiHeadAttention)')
    logger.info('=' * 80)

    inputs = []
    
    # NEW: ChemBERTa Inputs for Cross-Attention
    chem_emb_layer = None
    if 'chemberta_ligand' in feature_dims:
        chem_lig_input = Input(shape=(feature_dims['chemberta_ligand'],), name='chemberta_ligand_input')
        chem_mut_input = Input(shape=(feature_dims['chemberta_mutation'],), name='chemberta_mutation_input')
        inputs.extend([chem_lig_input, chem_mut_input])
        
        # Cross-Attention Logic
        # 1. Reshape to (Batch, SeqLen=1, Dim=768)
        lig_seq = Reshape((1, feature_dims['chemberta_ligand']))(chem_lig_input)
        mut_seq = Reshape((1, feature_dims['chemberta_mutation']))(chem_mut_input)
        
        # 2. MultiHeadAttention (Query=Ligand, Value=Key=Mutation)
        # 768 dim / 8 heads = 96 key_dim
        attn_out = MultiHeadAttention(num_heads=8, key_dim=96)(query=lig_seq, value=mut_seq)
        attn_out = Dropout(0.1)(attn_out)
        
        # 3. Residual connection + Layer Normalization
        # Add (Ligand + Attention) -> Norm
        res_out = Add()([lig_seq, attn_out])
        norm_out = LayerNormalization(epsilon=1e-6)(res_out)
        
        # 4. Flatten back to (Batch, 768)
        chem_context = Flatten()(norm_out)
        
        # 5. Connect to Chem Branch
        chem_branch = Dense(32, kernel_initializer='he_normal', name='chem_dense1')(chem_context)
        chem_branch = LeakyReLU(alpha=0.1, name='chem_leaky1')(chem_branch)
        chem_branch = BatchNormalization(name='chem_bn1')(chem_branch)
        chem_emb = Dense(4, activation='tanh', name='chem_embedding')(chem_branch)
        chem_emb_layer = chem_emb

    final_interaction_input = Input(shape=(feature_dims['final_fp_interaction'],), name='final_fp_interaction')
    lig_mut_mix_inter_intra_input = Input(shape=(feature_dims['lig_mut_mix_inter_intra'],), name='lig_mut_mix_inter_intra')
    inter_interaction_input = Input(shape=(feature_dims['inter_interaction'],), name='inter_interaction')
    intra_interaction_input = Input(shape=(feature_dims['intra_interaction'],), name='intra_interaction')
    mut_inter_input = Input(shape=(feature_dims['mut_inter'],), name='mut_inter')
    lig_inter_input = Input(shape=(feature_dims['lig_inter'],), name='lig_inter')
    mut_intra_input = Input(shape=(feature_dims['mut_intra'],), name='mut_intra')
    lig_intra_input = Input(shape=(feature_dims['lig_intra'],), name='lig_intra')
    
    inputs.extend([
        final_interaction_input, lig_mut_mix_inter_intra_input, inter_interaction_input, intra_interaction_input,
        mut_inter_input, lig_inter_input, mut_intra_input, lig_intra_input
    ])

    final_branch = Dense(32, kernel_initializer='he_normal', name='final_dense1')(final_interaction_input)
    final_branch = LeakyReLU(alpha=0.1, name='final_leaky1')(final_branch)
    final_branch = BatchNormalization(name='final_bn1')(final_branch)
    final_branch = Dropout(0.1, name='final_dropout1')(final_branch)
    final_branch = Dense(16, kernel_initializer='he_normal', name='final_dense2')(final_branch)
    final_branch = LeakyReLU(alpha=0.1, name='final_leaky2')(final_branch)
    final_emb = Dense(8, activation='tanh', name='final_embedding')(final_branch)

    mix_branch = Dense(8, kernel_initializer='he_normal', name='mix_inter_intra_dense')(lig_mut_mix_inter_intra_input)
    mix_branch = LeakyReLU(alpha=0.1, name='mix_inter_intra_leaky')(mix_branch)
    mix_branch_emb = Dense(4, activation='tanh', name='mix_inter_intra_embedding')(mix_branch)

    if chem_emb_layer is not None:
        priority1_combined = Concatenate(name='priority1_combined')([final_emb, mix_branch_emb, chem_emb_layer])
    else:
        priority1_combined = Concatenate(name='priority1_combined')([final_emb, mix_branch_emb])

    inter_interact_branch = Dense(48, kernel_initializer='he_normal', name='inter_dense1')(inter_interaction_input)
    inter_interact_branch = LeakyReLU(alpha=0.1, name='inter_leaky1')(inter_interact_branch)
    inter_interact_branch = BatchNormalization(name='inter_bn1')(inter_interact_branch)
    inter_gate = Dense(48, activation='sigmoid', kernel_initializer='glorot_uniform', name='inter_gate')(priority1_combined)
    inter_gated = Multiply(name='inter_gating')([inter_interact_branch, inter_gate])
    inter_gated = Dropout(0.1, name='inter_dropout')(inter_gated)
    inter_branch = Dense(24, kernel_initializer='he_normal', name='inter_dense2')(inter_gated)
    inter_branch = LeakyReLU(alpha=0.1, name='inter_leaky2')(inter_branch)
    inter_emb = Dense(12, activation='tanh', name='inter_embedding')(inter_branch)

    priority1_2_combined = Concatenate(name='priority1_2_combined')([priority1_combined, inter_emb])

    intra_interact_branch = Dense(48, kernel_initializer='he_normal', name='intra_dense1')(intra_interaction_input)
    intra_interact_branch = LeakyReLU(alpha=0.1, name='intra_leaky1')(intra_interact_branch)
    intra_interact_branch = BatchNormalization(name='intra_bn1')(intra_interact_branch)
    intra_gate = Dense(48, activation='sigmoid', kernel_initializer='glorot_uniform', name='intra_gate')(priority1_2_combined)
    intra_gated = Multiply(name='intra_gating')([intra_interact_branch, intra_gate])
    intra_gated = Dropout(0.1, name='intra_dropout')(intra_gated)
    intra_branch = Dense(24, kernel_initializer='he_normal', name='intra_dense2')(intra_gated)
    intra_branch = LeakyReLU(alpha=0.1, name='intra_leaky2')(intra_branch)
    intra_emb = Dense(12, activation='tanh', name='intra_embedding')(intra_branch)

    priority1_2_3_combined = Concatenate(name='priority1_2_3_combined')([priority1_2_combined, intra_emb])

    mut_inter_branch = Dense(32, kernel_initializer='he_normal', name='mut_inter_dense1')(mut_inter_input)
    mut_inter_branch = LeakyReLU(alpha=0.1, name='mut_inter_leaky1')(mut_inter_branch)
    mut_inter_branch = BatchNormalization(name='mut_inter_bn')(mut_inter_branch)
    mut_inter_gate = Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform', name='mut_inter_gate')(priority1_2_3_combined)
    mut_inter_gated = Multiply(name='mut_inter_gating')([mut_inter_branch, mut_inter_gate])
    mut_inter_gated = Dropout(0.1, name='mut_inter_dropout')(mut_inter_gated)
    mut_inter_branch = Dense(16, kernel_initializer='he_normal', name='mut_inter_dense2')(mut_inter_gated)
    mut_inter_branch = LeakyReLU(alpha=0.1, name='mut_inter_leaky2')(mut_inter_branch)
    mut_inter_emb = Dense(8, activation='tanh', name='mut_inter_embedding')(mut_inter_branch)

    lig_inter_branch = Dense(32, kernel_initializer='he_normal', name='lig_inter_dense1')(lig_inter_input)
    lig_inter_branch = LeakyReLU(alpha=0.1, name='lig_inter_leaky1')(lig_inter_branch)
    lig_inter_branch = BatchNormalization(name='lig_inter_bn')(lig_inter_branch)
    lig_inter_gate = Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform', name='lig_inter_gate')(priority1_2_3_combined)
    lig_inter_gated = Multiply(name='lig_inter_gating')([lig_inter_branch, lig_inter_gate])
    lig_inter_gated = Dropout(0.1, name='lig_inter_dropout')(lig_inter_gated)
    lig_inter_branch = Dense(16, kernel_initializer='he_normal', name='lig_inter_dense2')(lig_inter_gated)
    lig_inter_branch = LeakyReLU(alpha=0.1, name='lig_inter_leaky2')(lig_inter_branch)
    lig_inter_emb = Dense(8, activation='tanh', name='lig_inter_embedding')(lig_inter_branch)

    inter_combined = Concatenate(name='inter_combined')([mut_inter_emb, lig_inter_emb])

    priority1_to_5_combined = Concatenate(name='priority1_to_5_combined')([priority1_2_3_combined, inter_combined])

    mut_intra_branch = Dense(32, kernel_initializer='he_normal', name='mut_intra_dense1')(mut_intra_input)
    mut_intra_branch = LeakyReLU(alpha=0.1, name='mut_intra_leaky1')(mut_intra_branch)
    mut_intra_branch = BatchNormalization(name='mut_intra_bn')(mut_intra_branch)
    mut_intra_gate = Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform', name='mut_intra_gate')(priority1_to_5_combined)
    mut_intra_gated = Multiply(name='mut_intra_gating')([mut_intra_branch, mut_intra_gate])
    mut_intra_gated = Dropout(0.25, name='mut_intra_dropout')(mut_intra_gated)
    mut_intra_branch = Dense(16, kernel_initializer='he_normal', name='mut_intra_dense2')(mut_intra_gated)
    mut_intra_branch = LeakyReLU(alpha=0.1, name='mut_intra_leaky2')(mut_intra_branch)
    mut_intra_emb = Dense(8, activation='tanh', name='mut_intra_embedding')(mut_intra_branch)

    lig_intra_branch = Dense(32, kernel_initializer='he_normal', name='lig_intra_dense1')(lig_intra_input)
    lig_intra_branch = LeakyReLU(alpha=0.1, name='lig_intra_leaky1')(lig_intra_branch)
    lig_intra_branch = BatchNormalization(name='lig_intra_bn')(lig_intra_branch)
    lig_intra_gate = Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform', name='lig_intra_gate')(priority1_to_5_combined)
    lig_intra_gated = Multiply(name='lig_intra_gating')([lig_intra_branch, lig_intra_gate])
    lig_intra_gated = Dropout(0.25, name='lig_intra_dropout')(lig_intra_gated)
    lig_intra_branch = Dense(16, kernel_initializer='he_normal', name='lig_intra_dense2')(lig_intra_gated)
    lig_intra_branch = LeakyReLU(alpha=0.1, name='lig_intra_leaky2')(lig_intra_branch)
    lig_intra_emb = Dense(8, activation='tanh', name='lig_intra_embedding')(lig_intra_branch)

    intra_combined = Concatenate(name='intra_combined')([mut_intra_emb, lig_intra_emb])

    all_combined = Concatenate(name='all_combined')([priority1_2_3_combined, inter_combined, intra_combined])
    x = Dense(128, kernel_initializer='he_normal', name='integration_dense1')(all_combined)
    x = LeakyReLU(alpha=0.1, name='integration_leaky1')(x)
    x = BatchNormalization(name='integration_bn1')(x)
    x = Dropout(0.3, name='integration_dropout1')(x)
    x = Dense(64, kernel_initializer='he_normal', name='integration_dense2')(x)
    x = LeakyReLU(alpha=0.1, name='integration_leaky2')(x)
    x = BatchNormalization(name='integration_bn2')(x)
    x = Dropout(0.2, name='integration_dropout2')(x)
    x = Dense(32, kernel_initializer='he_normal', name='integration_dense3')(x)
    x = LeakyReLU(alpha=0.1, name='integration_leaky3')(x)

    embedding_layer = Dense(16, kernel_initializer='he_normal', name='embedding_layer')(x)
    embedding_output = LeakyReLU(alpha=0.1, name='embedding_output')(embedding_layer)

    activity_head = Dense(8, kernel_initializer='he_normal', name='activity_head')(embedding_output)
    activity_head = LeakyReLU(alpha=0.1, name='activity_head_activation')(activity_head)
    activity_head = Dropout(0.2, name='activity_head_dropout')(activity_head)
    activity_output = Dense(1, activation='linear', kernel_initializer='glorot_uniform', name='activity_output')(activity_head)

    docking_head = Dense(8, kernel_initializer='he_normal', name='docking_head')(embedding_output)
    docking_head = LeakyReLU(alpha=0.1, name='docking_head_activation')(docking_head)
    docking_head = Dropout(0.2, name='docking_head_dropout')(docking_head)
    docking_output = Dense(1, activation='linear', kernel_initializer='glorot_uniform', name='docking_output')(docking_head)

    model = Model(inputs=inputs, outputs=[activity_output, docking_output], name='priority_hierarchical_model')
    model.compile(
        optimizer=Adam(learning_rate=0.003),
        loss={'activity_output': 'mean_squared_error', 'docking_output': 'mean_squared_error'},
        loss_weights={'activity_output': 1.0, 'docking_output': 0.6},
        metrics={'activity_output': ['mae', 'mse'], 'docking_output': ['mae', 'mse']}
    )
    return model

def build_rnn_sequential_model(embedding_dim, n_timesteps=6):
    sequence_input = Input(shape=(n_timesteps, embedding_dim), name='mutation_sequence')
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), name='bilstm_1')(sequence_input)
    lstm_out = BatchNormalization(name='bn_lstm1')(lstm_out)
    lstm_out = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2), name='bilstm_2')(lstm_out)
    lstm_out = BatchNormalization(name='bn_lstm2')(lstm_out)
    gru_out = Bidirectional(GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), name='bigru_1')(sequence_input)
    gru_out = BatchNormalization(name='bn_gru1')(gru_out)
    gru_out = Bidirectional(GRU(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2), name='bigru_2')(gru_out)
    gru_out = BatchNormalization(name='bn_gru2')(gru_out)
    combined = Concatenate(name='lstm_gru_combined')([lstm_out, gru_out])
    x = Dense(128, activation='relu', name='rnn_dense1')(combined)
    x = BatchNormalization(name='rnn_bn1')(x)
    x = Dropout(0.3, name='rnn_dropout1')(x)
    x = Dense(64, activation='relu', name='rnn_dense2')(x)
    x = BatchNormalization(name='rnn_bn2')(x)
    x = Dropout(0.2, name='rnn_dropout2')(x)
    x = Dense(32, activation='relu', name='rnn_dense3')(x)
    x = Dropout(0.1, name='rnn_dropout3')(x)
    activity_final = Dense(16, activation='relu', name='activity_final_head')(x)
    activity_final = Dropout(0.15, name='activity_final_dropout')(activity_final)
    activity_output = Dense(1, activation='linear', name='final_activity_output')(activity_final)
    docking_final = Dense(16, activation='relu', name='docking_final_head')(x)
    docking_final = Dropout(0.15, name='docking_final_dropout')(docking_final)
    docking_output = Dense(1, activation='linear', name='final_docking_output')(docking_final)
    model = Model(inputs=sequence_input, outputs=[activity_output, docking_output], name='rnn_sequential_model')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'final_activity_output': 'mean_squared_error', 'final_docking_output': 'mean_squared_error'},
        loss_weights={'final_activity_output': 1.0, 'final_docking_output': 0.7},
        metrics={'final_activity_output': ['mae', 'mse'], 'final_docking_output': ['mae', 'mse']}
    )
    return model

def keras_main():
    device = get_device()
    print('\nLoading ChemBERTa model for embedding extraction...')
    tokenizer, chem_model, device = load_chemberta(device=device)
    
    print('Computing ChemBERTa embeddings for training ligands and mutation sites...')
    lig_smiles_list = ligand_smiles_valid.astype(str).tolist()
    lig_embs = get_chemberta_embeddings(lig_smiles_list, tokenizer, chem_model, device)

    site_smiles_lists = [
        full_smiles_valid.astype(str).tolist(),
        mutation_smiles_valid.astype(str).tolist(),
        mut_hinge_p_loop_valid.astype(str).tolist(),
        mut_helix_valid.astype(str).tolist(),
        mut_dfg_a_loop_valid.astype(str).tolist(),
        mut_hrd_cat_valid.astype(str).tolist(),
    ]
    site_embs_list = []
    for site_list in site_smiles_lists:
        site_embs_list.append(get_chemberta_embeddings(site_list, tokenizer, chem_model, device))

    mutation_sites = [
        ('FULL_SMILES', full_smiles_valid),
        ('ATP_POCKET', mutation_smiles_valid),
        ('P_LOOP_HINGE', mut_hinge_p_loop_valid),
        ('C_HELIX', mut_helix_valid),
        ('DFG_A_LOOP', mut_dfg_a_loop_valid),
        ('HRD_CAT', mut_hrd_cat_valid),
    ]

    all_feature_dicts = []
    all_valid_indices = []

    for site_name, mut_smiles_series in mutation_sites:
        print(f"\n{'='*80}\nProcessing {site_name}\n{'='*80}")
        feature_dict = generate_hierarchical_features(ligand_smiles_valid, mut_smiles_series)
        all_feature_dicts.append(feature_dict)
        all_valid_indices.append(set(feature_dict['valid_indices']))

    common_valid_indices = set.intersection(*all_valid_indices)
    common_valid_indices = sorted(list(common_valid_indices))
    print(f"\nCommon valid samples across all sites: {len(common_valid_indices)}")
    if len(common_valid_indices) == 0:
        print('\n✖ ERROR: No samples remain after filtering!')
        sys.exit(1)

    for i, feature_dict in enumerate(all_feature_dicts):
        site_valid_idx = feature_dict['valid_indices']
        mask = np.isin(site_valid_idx, common_valid_indices)
        for key in ['lig_inter', 'mut_inter', 'inter_interaction', 'lig_intra', 'mut_intra', 'intra_interaction', 'lig_mut_mix_inter_intra', 'final_fp_interaction']:
            all_feature_dicts[i][key] = feature_dict[key][mask]

    y_train1 = activity_values_valid.iloc[common_valid_indices].values
    y_train1 = np.log1p(y_train1)
    y_scaler1 = StandardScaler()
    y_train_scaled1 = y_scaler1.fit_transform(y_train1.reshape(-1, 1)).flatten()
    y_train2 = activity_values2_valid[common_valid_indices]
    y_scaler2 = StandardScaler()
    y_train_scaled2 = y_scaler2.fit_transform(y_train2.reshape(-1, 1)).flatten()

    all_scalers = []
    # No "pure" chem scalers here, as we scale ChemBERTa inside or before training. 
    # We will standardize ChemBERTa embeddings before passing to network for stability.
    chem_emb_scaler = StandardScaler()
    # Fit on all available embeddings to get robust mean/std
    all_embs_concat = np.concatenate([lig_embs] + site_embs_list, axis=0)
    chem_emb_scaler.fit(all_embs_concat)
    
    # Scale all embeddings
    lig_embs = chem_emb_scaler.transform(lig_embs)
    for i in range(len(site_embs_list)):
        site_embs_list[i] = chem_emb_scaler.transform(site_embs_list[i])

    all_embeddings = []
    all_site_histories = []

    for site_idx, (site_name, _) in enumerate(mutation_sites):
        print(f"\n{'='*80}\nSite {site_idx+1}/6: {site_name}\n{'='*80}")
        feature_dict = all_feature_dicts[site_idx]

        scalers = {}
        scaled_features = {}
        for key in ['lig_inter', 'mut_inter', 'inter_interaction', 'lig_intra', 'mut_intra', 'intra_interaction', 'lig_mut_mix_inter_intra', 'final_fp_interaction']:
            scalers[key] = StandardScaler()
            scaled_features[key] = scalers[key].fit_transform(feature_dict[key])
        all_scalers.append(scalers)

        # Prepare ChemBERTa Inputs
        chem_lig_common = lig_embs[common_valid_indices]  # (N, 768)
        chem_mut_common = site_embs_list[site_idx][common_valid_indices]  # (N, 768)

        feature_dims = {
            'lig_inter': scaled_features['lig_inter'].shape[1],
            'mut_inter': scaled_features['mut_inter'].shape[1],
            'inter_interaction': scaled_features['inter_interaction'].shape[1],
            'lig_intra': scaled_features['lig_intra'].shape[1],
            'mut_intra': scaled_features['mut_intra'].shape[1],
            'intra_interaction': scaled_features['intra_interaction'].shape[1],
            'lig_mut_mix_inter_intra': scaled_features['lig_mut_mix_inter_intra'].shape[1],
            'final_fp_interaction': scaled_features['final_fp_interaction'].shape[1],
            'chemberta_ligand': 768,
            'chemberta_mutation': 768,
        }

        model = build_priority_hierarchical_model(feature_dims)
        
        checkpoint = ModelCheckpoint(f'hierarchical_model_{site_name}.h5', monitor='val_loss', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)

        print(f"\nTraining {site_name} model...")
        history = model.fit(
            x=[
                chem_lig_common, # Ligand Embedding
                chem_mut_common, # Mutation Embedding
                scaled_features['final_fp_interaction'],
                scaled_features['lig_mut_mix_inter_intra'],
                scaled_features['inter_interaction'],
                scaled_features['intra_interaction'],
                scaled_features['mut_inter'],
                scaled_features['lig_inter'],
                scaled_features['mut_intra'],
                scaled_features['lig_intra'],
            ],
            y={'activity_output': y_train_scaled1, 'docking_output': y_train_scaled2},
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop, checkpoint],
            verbose=1,
        )

        try:
            all_site_histories.append(history.history)
        except Exception:
            all_site_histories.append({'loss': [], 'val_loss': []})

        embedding_model = Model(inputs=model.inputs, outputs=model.get_layer('embedding_output').output)
        embeddings = embedding_model.predict([
            chem_lig_common,
            chem_mut_common,
            scaled_features['final_fp_interaction'],
            scaled_features['lig_mut_mix_inter_intra'],
            scaled_features['inter_interaction'],
            scaled_features['intra_interaction'],
            scaled_features['mut_inter'],
            scaled_features['lig_inter'],
            scaled_features['mut_intra'],
            scaled_features['lig_intra'],
        ], verbose=0)

        all_embeddings.append(embeddings)

    sequential_embeddings = np.stack(all_embeddings, axis=1)
    rnn_model = build_rnn_sequential_model(sequential_embeddings.shape[2], sequential_embeddings.shape[1])
    
    rnn_checkpoint = ModelCheckpoint('rnn_sequential_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    rnn_early_stop = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1)

    print('\nTraining RNN-LSTM model...')
    rnn_history = rnn_model.fit(
        x=sequential_embeddings,
        y={'final_activity_output': y_train_scaled1, 'final_docking_output': y_train_scaled2},
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[rnn_early_stop, rnn_checkpoint],
        verbose=1,
    )

    with open('feature_scalers.pkl', 'wb') as f:
        pickle.dump(all_scalers, f)
    with open('y_scalers.pkl', 'wb') as f:
        pickle.dump({'y_scaler1': y_scaler1, 'y_scaler2': y_scaler2}, f)
    with open('chembert_scalers.pkl', 'wb') as f:
        pickle.dump([chem_emb_scaler], f) # Save the single scaler used for all

    unique_mutation_profiles.to_csv('mutation_profiles.csv', index=False)
    
    # Prediction loop for Control/Drug
    all_control_results = []
    
    for profile_idx, profile_row in unique_mutation_profiles.iterrows():
        mutation_name = profile_row['tkd']
        mut_site_smiles = [
            profile_row['smiles_full_egfr'],
            profile_row['smiles 718_862_atp_pocket'],
            profile_row['smiles_p_loop'],
            profile_row['smiles_c_helix'],
            profile_row['smiles_l858r_a_loop_dfg_motif'],
            profile_row['smiles_catalytic_hrd_motif'],
        ]
        
        control_embeddings_all_sites = []
        control_valid_idx = None

        for site_idx, (site_name, _) in enumerate(mutation_sites):
            mut_smi_site = mut_site_smiles[site_idx]
            mut_inter = generate_mut_inter_features(mut_smi_site)
            mut_intra = generate_mut_intra_features(mut_smi_site)
            if mut_inter is None or mut_intra is None: continue

            control_features = {k: [] for k in ['lig_inter', 'mut_inter', 'inter_interaction', 'lig_intra', 'mut_intra', 'intra_interaction', 'lig_mut_mix_inter_intra', 'final_fp_interaction']}
            site_control_valid_idx = []

            for idx, control_smi in enumerate(control_smiles):
                lig_inter = generate_lig_inter_features(control_smi)
                lig_intra = generate_lig_intra_features(control_smi)
                if lig_inter is None or lig_intra is None: continue
                
                lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra = generate_custom_features(lig_inter, mut_inter, lig_intra, mut_intra)
                inter_interaction = generate_inter_interaction_features(lig_inter, mut_inter)
                intra_interaction = generate_intra_interaction_features(lig_intra, mut_intra)
                if len(lig_mut_inter) > 0: inter_interaction = np.concatenate([np.array(lig_mut_inter), inter_interaction])
                if len(lig_mut_intra) > 0: intra_interaction = np.concatenate([np.array(lig_mut_intra), intra_interaction])
                final_fp_interaction = generate_final_interaction_features(control_smi, mut_smi_site)
                
                control_features['lig_inter'].append(lig_inter)
                control_features['mut_inter'].append(mut_inter)
                control_features['inter_interaction'].append(inter_interaction)
                control_features['lig_intra'].append(lig_intra)
                control_features['mut_intra'].append(mut_intra)
                control_features['intra_interaction'].append(intra_interaction)
                control_features['lig_mut_mix_inter_intra'].append(np.array(lig_mut_mix_inter_intra))
                control_features['final_fp_interaction'].append(final_fp_interaction)
                site_control_valid_idx.append(idx)

            if len(site_control_valid_idx) == 0: break
            if control_valid_idx is None: control_valid_idx = site_control_valid_idx

            scaled_control = {}
            scalers = all_scalers[site_idx]
            for key in control_features.keys():
                scaled_control[key] = scalers[key].transform(np.array(control_features[key]))

            control_lig_emb = get_chemberta_embeddings(control_smiles.astype(str).tolist(), tokenizer, chem_model, device)
            mut_emb_site = get_chemberta_embeddings([mut_smi_site] * len(control_lig_emb), tokenizer, chem_model, device)
            
            # Scale
            control_lig_emb = chem_emb_scaler.transform(control_lig_emb)
            mut_emb_site = chem_emb_scaler.transform(mut_emb_site)

            hierarchical_model = load_model(f'hierarchical_model_{site_name}.h5', compile=False)
            # Recompile isn't strictly needed for predict but good practice
            embedding_model = Model(inputs=hierarchical_model.inputs, outputs=hierarchical_model.get_layer('embedding_output').output)
            
            site_embeddings = embedding_model.predict([
                control_lig_emb,
                mut_emb_site,
                scaled_control['final_fp_interaction'],
                scaled_control['lig_mut_mix_inter_intra'],
                scaled_control['inter_interaction'],
                scaled_control['intra_interaction'],
                scaled_control['mut_inter'],
                scaled_control['lig_inter'],
                scaled_control['mut_intra'],
                scaled_control['lig_intra'],
            ], verbose=0)
            
            control_embeddings_all_sites.append(site_embeddings)

        if len(control_embeddings_all_sites) == 6 and control_valid_idx is not None:
            control_sequential = np.stack(control_embeddings_all_sites, axis=1)
            predictions = rnn_model.predict(control_sequential, verbose=0)
            act_pred = np.expm1(y_scaler1.inverse_transform(predictions[0].flatten().reshape(-1, 1)).flatten())
            dock_pred = y_scaler2.inverse_transform(predictions[1].flatten().reshape(-1, 1)).flatten()
            
            for idx, ap, dp in zip(control_valid_idx, act_pred, dock_pred):
                all_control_results.append({'mutation_name': mutation_name, 'control_name': control_name.iloc[idx], 'compound_smiles': control_smiles.iloc[idx], 'predicted_activity': ap, 'predicted_docking': dp})

    df_control_results = pd.DataFrame(all_control_results)
    # df_control_results.to_csv('control_predictions_rnn_chembert.csv', index=False)  <-- REMOVED early save

    print('Starting drug predictions...')
    all_drug_results = []
    
    for profile_idx, profile_row in unique_mutation_profiles.iterrows():
        mutation_name = profile_row['tkd']
        mut_site_smiles = [
            profile_row['smiles_full_egfr'],
            profile_row['smiles 718_862_atp_pocket'],
            profile_row['smiles_p_loop'],
            profile_row['smiles_c_helix'],
            profile_row['smiles_l858r_a_loop_dfg_motif'],
            profile_row['smiles_catalytic_hrd_motif'],
        ]
        
        drug_embeddings_all_sites = []
        drug_valid_idx = None
        
        for site_idx, (site_name, _) in enumerate(mutation_sites):
            mut_smi_site = mut_site_smiles[site_idx]
            mut_inter = generate_mut_inter_features(mut_smi_site)
            mut_intra = generate_mut_intra_features(mut_smi_site)
            if mut_inter is None or mut_intra is None: continue
            
            drug_features = {k: [] for k in ['lig_inter', 'mut_inter', 'inter_interaction', 'lig_intra', 'mut_intra', 'intra_interaction', 'lig_mut_mix_inter_intra', 'final_fp_interaction']}
            site_drug_valid_idx = []
            
            # Note: For drugs, we iterate over all drugs for EACH mutation profile
            for idx, drug_smi in enumerate(drug_smiles):
                lig_inter = generate_lig_inter_features(drug_smi)
                lig_intra = generate_lig_intra_features(drug_smi)
                if lig_inter is None or lig_intra is None: continue
                
                lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra = generate_custom_features(lig_inter, mut_inter, lig_intra, mut_intra)
                inter_interaction = generate_inter_interaction_features(lig_inter, mut_inter)
                intra_interaction = generate_intra_interaction_features(lig_intra, mut_intra)
                if len(lig_mut_inter) > 0: inter_interaction = np.concatenate([np.array(lig_mut_inter), inter_interaction])
                if len(lig_mut_intra) > 0: intra_interaction = np.concatenate([np.array(lig_mut_intra), intra_interaction])
                final_fp_interaction = generate_final_interaction_features(drug_smi, mut_smi_site)
                
                drug_features['lig_inter'].append(lig_inter)
                drug_features['mut_inter'].append(mut_inter)
                drug_features['inter_interaction'].append(inter_interaction)
                drug_features['lig_intra'].append(lig_intra)
                drug_features['mut_intra'].append(mut_intra)
                drug_features['intra_interaction'].append(intra_interaction)
                drug_features['lig_mut_mix_inter_intra'].append(np.array(lig_mut_mix_inter_intra))
                drug_features['final_fp_interaction'].append(final_fp_interaction)
                site_drug_valid_idx.append(idx)
                
            if len(site_drug_valid_idx) == 0: break
            if drug_valid_idx is None: drug_valid_idx = site_drug_valid_idx
            
            scaled_drug = {}
            scalers = all_scalers[site_idx]
            for key in drug_features.keys():
                scaled_drug[key] = scalers[key].transform(np.array(drug_features[key]))
                
            drug_lig_emb = get_chemberta_embeddings(drug_smiles.astype(str).tolist(), tokenizer, chem_model, device)
            mut_emb_site = get_chemberta_embeddings([mut_smi_site] * len(drug_lig_emb), tokenizer, chem_model, device)
            
            drug_lig_emb = chem_emb_scaler.transform(drug_lig_emb)
            mut_emb_site = chem_emb_scaler.transform(mut_emb_site)

            hierarchical_model = load_model(f'hierarchical_model_{site_name}.h5', compile=False)
            embedding_model = Model(inputs=hierarchical_model.inputs, outputs=hierarchical_model.get_layer('embedding_output').output)
            
            site_embeddings = embedding_model.predict([
                drug_lig_emb,
                mut_emb_site,
                scaled_drug['final_fp_interaction'],
                scaled_drug['lig_mut_mix_inter_intra'],
                scaled_drug['inter_interaction'],
                scaled_drug['intra_interaction'],
                scaled_drug['mut_inter'],
                scaled_drug['lig_inter'],
                scaled_drug['mut_intra'],
                scaled_drug['lig_intra'],
            ], verbose=0)
            
            drug_embeddings_all_sites.append(site_embeddings)
            
        if len(drug_embeddings_all_sites) == 6 and drug_valid_idx is not None:
            drug_sequential = np.stack(drug_embeddings_all_sites, axis=1)
            predictions = rnn_model.predict(drug_sequential, verbose=0)
            act_pred = np.expm1(y_scaler1.inverse_transform(predictions[0].flatten().reshape(-1, 1)).flatten())
            dock_pred = y_scaler2.inverse_transform(predictions[1].flatten().reshape(-1, 1)).flatten()
            
            for idx, ap, dp in zip(drug_valid_idx, act_pred, dock_pred):
                all_drug_results.append({'mutation_name': mutation_name, 'compound_smiles': drug_smiles.iloc[idx], 'predicted_activity': ap, 'predicted_docking': dp})
                
    df_drug_results = pd.DataFrame(all_drug_results)
    # df_drug_results.to_csv('drug_predictions_rnn_chembert.csv', index=False) <-- REMOVED early save
    
    # Save results (Consolidated)
    df_control_results.to_csv('control_predictions_rnn_chembert.csv', index=False)
    df_drug_results.to_csv('drug_predictions_rnn_chembert.csv', index=False)
    print('Execution complete. Outputs saved.')

    # --- Training history plotting (Added per request) ---
    print('Generating training history plots...')
    try:
        plt.figure(figsize=(14, 6))

        # Left: hierarchical per-site losses
        plt.subplot(1, 2, 1)
        for i, h in enumerate(all_site_histories):
            # h is a dict (history.history)
            epochs = range(1, len(h.get('loss', [])) + 1)
            train_loss = h.get('loss', [])
            val_loss = h.get('val_loss', [])
            site_name = mutation_sites[i][0] if i < len(mutation_sites) else f'Site_{i+1}'
            
            if train_loss:
                plt.plot(epochs, train_loss, label=f'{site_name} train')
            if val_loss:
                plt.plot(epochs, val_loss, linestyle='--', label=f'{site_name} val')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Hierarchical per-site training (Keras MultiHead)')
        plt.legend(fontsize='small', loc='upper right')

        # Right: RNN training loss
        plt.subplot(1, 2, 2)
        # rnn_history is a History object, so we access .history
        r_hist = rnn_history.history
        r_epochs = range(1, len(r_hist.get('loss', [])) + 1)
        
        if r_hist.get('loss'):
            plt.plot(r_epochs, r_hist['loss'], label='RNN train loss')
        if r_hist.get('val_loss'):
            plt.plot(r_epochs, r_hist['val_loss'], linestyle='--', label='RNN val loss')
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('RNN training (Sequential)')
        plt.legend()

        plt.tight_layout()
        out_png = 'training_history_keras_crossattn.png'
        plt.savefig(out_png, dpi=200)
        print(f'Saved training history plot to {out_png}')
    except Exception as e:
        print('Could not generate training plot:', e)

if __name__ == '__main__':
    keras_main()
