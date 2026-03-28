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
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU for both TensorFlow and PyTorch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# FIX 1: Removed duplicate import blocks (os/sys/pickle/numpy/pandas/matplotlib
#         were each imported 2-3 times). Single clean import block below.
# FIX 2: Added missing Keras layer imports required to load models that contain
#         MultiHeadAttention, LayerNormalization, Add, Reshape and Flatten layers
#         (cross-attention block from corrected training script). Without these,
#         load_model() raises 'Unknown layer' errors.
# FIX 3: Added 'from transformers import AutoTokenizer, AutoModel' at module
#         level. These are used by load_chemberta() which is defined at module
#         scope; previously they were only imported inside make_predictions(),
#         so any call to load_chemberta() from outside that function would crash.

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel          # FIX 3

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Concatenate, Multiply,
    LSTM, GRU, Bidirectional, LeakyReLU,
    MultiHeadAttention, LayerNormalization, Add, Reshape, Flatten  # FIX 2
)
from tensorflow.keras.optimizers import Adam

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from rdkit import RDLogger
from numpy.linalg import norm

os.environ['MPLBACKEND'] = 'Agg'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(42)

# === Disk-backed Feature Cache ===
import hashlib
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', '.feature_cache')
_cache_hits = 0
_cache_misses = 0

def _get_cache_path(smiles):
    """Return the cache file path for a given SMILES string."""
    md5 = hashlib.md5(smiles.encode('utf-8')).hexdigest()
    return os.path.join(_CACHE_DIR, f'{md5}.pkl')

def _load_cached_features(smiles):
    """Load cached inter/intra features from disk. Returns (inter, intra) or None."""
    global _cache_hits, _cache_misses
    path = _get_cache_path(smiles)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            _cache_hits += 1
            return data['inter'], data['intra']
        except Exception:
            pass
    _cache_misses += 1
    return None

def _save_cached_features(smiles, inter, intra):
    """Save computed inter/intra features to disk cache."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _get_cache_path(smiles)
    try:
        with open(path, 'wb') as f:
            pickle.dump({'inter': inter, 'intra': intra}, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

def _generate_lig_features(smiles):
    """Compute both inter and intra features with a single MolFromSmiles call.
    Checks disk cache first; writes to disk cache on miss.
    Returns (inter_array, intra_array) or (None, None)."""
    cached = _load_cached_features(smiles)
    if cached is not None:
        return cached
    inter = generate_lig_inter_features(smiles)
    intra = generate_lig_intra_features(smiles)
    if inter is not None and intra is not None:
        _save_cached_features(smiles, inter, intra)
    return inter, intra


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
    
    c_linear1_size1 = safe_divide(lig_inter[6], lig_inter[14], default=0.0) * safe_divide(mut_inter[7], mut_inter[14], default=0.0)
    lig_mut_mix_inter_intra.append(c_linear1_size1)
    
    c_linear2_size1 = safe_divide(lig_inter[7], lig_inter[14], default=0.0) * safe_divide(mut_inter[6], mut_inter[14], default=0.0)
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
    
    kappa_ratio = safe_divide(lig_intra[21], mut_intra[21], default=0.0)
    lig_mut_intra.append(kappa_ratio)
    
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
            lig_inter, lig_intra = _generate_lig_features(lig_smi)
            ligand_cache[lig_smi] = (lig_inter, lig_intra)

        if mut_smi in mutation_cache:
            mut_inter, mut_intra = mutation_cache[mut_smi]
        else:
            mut_inter, mut_intra = _generate_lig_features(mut_smi)
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

def get_chemberta_embeddings(smiles, tokenizer, model, device, batch_size=32):
    """
    Generate ChemBERTa embeddings - MATCHES TRAINING SCRIPT
    Uses mean pooling (not CLS token)
    """
    # Pre-validate SMILES to avoid tokenization issues
    clean_smiles = []
    for s in smiles:
        if pd.isna(s) or str(s).strip() == '':
            clean_smiles.append('C')  # Use methane as fallback
        else:
            s_clean = str(s).strip()
            # Check if valid SMILES
            mol = Chem.MolFromSmiles(s_clean)
            if mol is None:
                print(f"Warning: Invalid SMILES '{s_clean}', using fallback")
                clean_smiles.append('C')
            else:
                clean_smiles.append(s_clean)
    
    # Tokenize with same parameters as training
    inputs = tokenizer(clean_smiles, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # CRITICAL: Validate and fix token IDs BEFORE creating dataset
    vocab_size = model.config.vocab_size
    max_token = input_ids.max().item()
    
    if max_token >= vocab_size:
        print(f"Warning: Found token ID {max_token} >= vocab_size ({vocab_size})")
        print(f"Problematic SMILES: {[s for i, s in enumerate(clean_smiles) if (input_ids[i] >= vocab_size).any()]}")
        # Replace invalid tokens with padding token
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        # Alternative: replace with UNK token
        # unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 3
        # input_ids = torch.where(input_ids >= vocab_size, unk_token_id, input_ids)
    
    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embs = []
    
    with torch.no_grad():
        for batch in loader:
            try:
                ids = batch[0].to(device)
                mask = batch[1].to(device)
                
                out = model(input_ids=ids, attention_mask=mask, return_dict=True)
                
                # Use mean pooling (same as training) NOT CLS token
                pooled = getattr(out, 'pooler_output', None)
                if pooled is None:
                    last = out.last_hidden_state
                    maskf = mask.unsqueeze(-1).float()
                    summed = (last * maskf).sum(1)
                    lengths = maskf.sum(1).clamp(min=1e-9)
                    pooled = summed / lengths
                embs.append(pooled.cpu())
            except RuntimeError as e:
                print(f"Error during embedding: {e}")
                print(f"Falling back to CPU for this batch")
                # Fallback: process on CPU
                ids_cpu = batch[0]
                mask_cpu = batch[1]
                model_cpu = model.cpu()
                out = model_cpu(input_ids=ids_cpu, attention_mask=mask_cpu, return_dict=True)
                pooled = getattr(out, 'pooler_output', None)
                if pooled is None:
                    last = out.last_hidden_state
                    maskf = mask_cpu.unsqueeze(-1).float()
                    summed = (last * maskf).sum(1)
                    lengths = maskf.sum(1).clamp(min=1e-9)
                    pooled = summed / lengths
                embs.append(pooled)
                model.to(device)  # Move back to original device
    
    embs = torch.cat(embs, dim=0)
    return embs.numpy()
# ============================================================================
# CONFIGURATION
# ============================================================================

# FIX 4: Corrected site names to match filenames saved by the training script.
# Training saves: hierarchical_model_FULL_SMILES.h5, hierarchical_model_P_LOOP_HINGE.h5
# Original had 'Full' and 'P_LOOP' which would cause FileNotFoundError.
MUTATION_SITES = ['FULL_SMILES', 'ATP_POCKET', 'P_LOOP_HINGE', 'C_HELIX', 'DFG_A_LOOP', 'HRD_CAT']
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
        chem_raw = pickle.load(f)
    # FIX 5: Training script saves pickle.dump([chem_emb_scaler], f) — a list
    # with one element. If we loaded from chembert_scalers.pkl we must index [0].
    # If it was saved as the bare scaler (chem_emb_scaler.pkl), it loads directly.
    if isinstance(chem_raw, list):
        chem_emb_scaler = chem_raw[0]
    else:
        chem_emb_scaler = chem_raw
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
    mutations = sorted(df_results['mutation'].unique())
    
    for mutation in mutations:
        mut_data = df_results[df_results['mutation'] == mutation]
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
        mut_data = df_results[df_results['mutation'] == mutation]
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
        mut_data = df_results[df_results['mutation'] == mutation]
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
        mut_data = df_results[df_results['mutation'] == mutation]
        
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
        mut_data = df_results[df_results['mutation'] == mutation]
        
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
    Make predictions using trained models
    
    Args:
        input_csv: Path to CSV with columns 'smiles', 'tkd' (mutation name)
                   Optional: 'standard value', 'dock' for evaluation
        model_dir: Directory containing trained models and scalers
        output_dir: Directory for output predictions
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    print("="*80)
    print("PREDICTION PIPELINE - ChemBERTa Cross-Attention")
    print("="*80)
    
    # ============================================================================
    # 1. DEFINE MUTATION SITES (must match training)
    # ============================================================================
    MUTATION_SITES = [
        'FULL_SMILES',
        'ATP_POCKET', 
        'P_LOOP_HINGE',
        'C_HELIX',
        'DFG_A_LOOP',
        'HRD_CAT'
    ]
    
    # ============================================================================
    # 2. INITIALIZE CHEMBERTA
    # ============================================================================
    print("\n[1/7] Initializing ChemBERTa...")
    device = torch.device('cpu')  # Will use CPU since CUDA_VISIBLE_DEVICES=''
    print(f"  Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    chemberta_model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)
    chemberta_model.eval()
    print("  ✓ ChemBERTa loaded")
    
    # ============================================================================
    # 3. LOAD MUTATION PROFILES
    # ============================================================================
    print("\n[2/7] Loading mutation profiles...")
    mutation_profile_path = os.path.join(model_dir, 'mutation_profiles.csv')
    if not os.path.exists(mutation_profile_path):
        print(f"  Warning: {mutation_profile_path} not found, trying training data...")
        mutation_profile_path = os.path.join(model_dir, 'tkd_descriptors_mutation.csv')
    
    df_mutation_profiles = pd.read_csv(mutation_profile_path, encoding='latin-1')
    print(f"  ✓ Loaded {len(df_mutation_profiles)} mutation profiles")
    
    # ============================================================================
    # 4. LOAD SCALERS
    # ============================================================================
    print("\n[3/7] Loading scalers...")
    
    # Feature scalers (one dict per site)
    with open(os.path.join(model_dir, 'feature_scalers.pkl'), 'rb') as f:
        all_scalers = pickle.load(f)
    print(f"  ✓ Loaded feature scalers for {len(all_scalers)} sites")
    
    # Y scalers
    with open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb') as f:
        y_scalers = pickle.load(f)
        y_scaler1 = y_scalers['y_scaler1']
        y_scaler2 = y_scalers['y_scaler2']
    print("  ✓ Loaded y scalers")
    
    # ChemBERTa embedding scaler
    with open(os.path.join(model_dir, 'chembert_scalers.pkl'), 'rb') as f:
        chem_scalers = pickle.load(f)
        chem_emb_scaler = chem_scalers[0]
    print("  ✓ Loaded ChemBERTa scaler")
    
    # ============================================================================
    # 5. LOAD HIERARCHICAL MODELS
    # ============================================================================
    print("\n[4/7] Loading hierarchical models...")
    hierarchical_models = {}
    for site_name in MUTATION_SITES:
        model_path = os.path.join(model_dir, f'hierarchical_model_{site_name}.h5')
        hierarchical_models[site_name] = load_model(model_path, compile=False)
        print(f"  ✓ Loaded {site_name}")
    
    # ============================================================================
    # 6. LOAD RNN MODEL
    # ============================================================================
    print("\n[5/7] Loading RNN sequential model...")
    rnn_model = load_model(os.path.join(model_dir, 'rnn_sequential_model.h5'), compile=False)
    print("  ✓ RNN model loaded")
    
    # ============================================================================
    # 7. LOAD INPUT DATA
    # ============================================================================
    print(f"\n[6/7] Loading input data from {input_csv}...")
    df_pred = pd.read_csv(input_csv, encoding='latin-1')
    print(f"  ✓ Loaded {len(df_pred)} samples")
    
    # Check for ground truth columns
    has_ground_truth = 'standard value' in df_pred.columns and 'dock' in df_pred.columns
    if has_ground_truth:
        print("  ✓ Ground truth detected - will evaluate predictions")
    
    # ============================================================================
    # 8. MAKE PREDICTIONS
    # ============================================================================
    print(f"\n[7/7] Making predictions...")
    results = []
    
    for pred_idx, pred_row in df_pred.iterrows():
        lig_smiles = pred_row['smiles']
        mutant_name = pred_row['tkd']
        
        # Validate SMILES
        if pd.isna(lig_smiles) or lig_smiles == '':
            print(f"Warning: Empty SMILES at row {pred_idx}, skipping")
            continue

        # Validate Mutation
        if pd.isna(mutant_name):
            print(f"Warning: Empty mutation at row {pred_idx}, skipping")
            continue
        
        # Find mutation profile
        mutation_profile = df_mutation_profiles[df_mutation_profiles['tkd'] == mutant_name]
        
        if len(mutation_profile) == 0:
            print(f"Warning: Mutation '{mutant_name}' not found in training data, skipping")
            continue
        
        mutation_profile = mutation_profile.iloc[0]
        
        # Get mutation SMILES for all sites (matching MUTATION_SITES order)
        mutation_smiles_list = [
            mutation_profile['smiles_full_egfr'],
            mutation_profile['smiles 718_862_atp_pocket'],
            mutation_profile['smiles_p_loop'],
            mutation_profile['smiles_c_helix'],
            mutation_profile['smiles_l858r_a_loop_dfg_motif'],
            mutation_profile['smiles_catalytic_hrd_motif']
        ]
        
        # Process through each hierarchical site
        embeddings_all_sites = []
        valid_site_processing = True
        
        for site_idx, (site_name, mutation_smiles) in enumerate(zip(MUTATION_SITES, mutation_smiles_list)):
            
            # Generate ChemBERTa embeddings
            lig_chem_emb = get_chemberta_embeddings([lig_smiles], tokenizer, chemberta_model, device, batch_size=1)
            mut_chem_emb = get_chemberta_embeddings([mutation_smiles], tokenizer, chemberta_model, device, batch_size=1)
            
            if lig_chem_emb is None or mut_chem_emb is None:
                print(f"Warning: Could not generate ChemBERTa embeddings for {mutant_name} at {site_name}, skipping")
                valid_site_processing = False
                break
            
            # Scale ChemBERTa embeddings
            lig_chem_emb_scaled = chem_emb_scaler.transform(lig_chem_emb)
            mut_chem_emb_scaled = chem_emb_scaler.transform(mut_chem_emb)
            
            # Generate physicochemical features
            lig_inter, lig_intra = _generate_lig_features(lig_smiles)
            mut_inter, mut_intra = _generate_lig_features(mutation_smiles)
            
            if any(f is None for f in [lig_inter, mut_inter, lig_intra, mut_intra]):
                print(f"Warning: Could not generate physicochemical features for {mutant_name} at {site_name}, skipping")
                valid_site_processing = False
                break
            
            # Generate interaction features
            lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra = generate_custom_features(
                lig_inter, mut_inter, lig_intra, mut_intra
            )
            
            inter_interaction = generate_inter_interaction_features(lig_inter, mut_inter)
            intra_interaction = generate_intra_interaction_features(lig_intra, mut_intra)
            
            # Concatenate custom features
            if len(lig_mut_inter) > 0:
                inter_interaction = np.concatenate([np.array(lig_mut_inter), inter_interaction])
            
            if len(lig_mut_intra) > 0:
                intra_interaction = np.concatenate([np.array(lig_mut_intra), intra_interaction])
            
            final_fp_interaction = generate_final_interaction_features(lig_smiles, mutation_smiles)
            
            # Scale physicochemical features
            scalers = all_scalers[site_idx]
            scaled_features = {
                'final_fp_interaction': scalers['final_fp_interaction'].transform(final_fp_interaction.reshape(1, -1)),
                'lig_mut_mix_inter_intra': scalers['lig_mut_mix_inter_intra'].transform(np.array(lig_mut_mix_inter_intra).reshape(1, -1)),
                'inter_interaction': scalers['inter_interaction'].transform(inter_interaction.reshape(1, -1)),
                'intra_interaction': scalers['intra_interaction'].transform(intra_interaction.reshape(1, -1)),
                'mut_inter': scalers['mut_inter'].transform(mut_inter.reshape(1, -1)),
                'lig_inter': scalers['lig_inter'].transform(lig_inter.reshape(1, -1)),
                'mut_intra': scalers['mut_intra'].transform(mut_intra.reshape(1, -1)),
                'lig_intra': scalers['lig_intra'].transform(lig_intra.reshape(1, -1))
            }
            
            # Get embedding from hierarchical model
            # Input order: [chemberta_ligand, chemberta_mutation, final_fp, lig_mut_mix, inter, intra, mut_inter, lig_inter, mut_intra, lig_intra]
            hierarchical_model = hierarchical_models[site_name]
            embedding_model = Model(
                inputs=hierarchical_model.inputs,
                outputs=hierarchical_model.get_layer('embedding_output').output
            )
            
            site_embedding = embedding_model.predict([
                lig_chem_emb_scaled,
                mut_chem_emb_scaled,
                scaled_features['final_fp_interaction'],
                scaled_features['lig_mut_mix_inter_intra'],
                scaled_features['inter_interaction'],
                scaled_features['intra_interaction'],
                scaled_features['mut_inter'],
                scaled_features['lig_inter'],
                scaled_features['mut_intra'],
                scaled_features['lig_intra']
            ], verbose=0)
            
            embeddings_all_sites.append(site_embedding)
        
        if not valid_site_processing:
            continue
        
        # Stack embeddings for RNN
        sequential_input = np.stack(embeddings_all_sites, axis=1)
        
        # Get final predictions from RNN
        predictions = rnn_model.predict(sequential_input, verbose=0)
        pred_activity_scaled = predictions[0].flatten()[0]
        pred_docking_scaled = predictions[1].flatten()[0]
        
        # Inverse transform
        pred_activity_log1p = y_scaler1.inverse_transform([[pred_activity_scaled]])[0, 0]
        pred_activity = np.expm1(pred_activity_log1p)
        pred_docking = y_scaler2.inverse_transform([[pred_docking_scaled]])[0, 0]
        
        # Store result
        res = {
            'smiles': lig_smiles,
            'mutation': mutant_name,
            'predicted_activity': pred_activity,
            'predicted_docking': pred_docking
        }
        
        if has_ground_truth:
            res['actual_activity'] = pred_row['standard value']
            res['actual_docking'] = pred_row['dock']
            
        # Keep other columns
        for col in df_pred.columns:
            if col not in res and col not in ['smiles', 'tkd', 'standard value', 'dock']:
                 res[col] = pred_row[col]
                 
        results.append(res)
        
        if (pred_idx + 1) % 10 == 0:
            print(f"  Processed {pred_idx + 1}/{len(df_pred)} samples")
    
    if not results:
        print("Error: No valid samples found to predict.")
        return None
        
    df_results = pd.DataFrame(results)
    
    # Save output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'predictions_adv_physchem_chemerta_crossattention2.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    # Evaluation (if ground truth exists)
    if has_ground_truth and len(df_results) > 0:
        evaluate_and_plot(df_results, output_dir, 'adv_physchem_chemerta_crossattention2')
        
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
