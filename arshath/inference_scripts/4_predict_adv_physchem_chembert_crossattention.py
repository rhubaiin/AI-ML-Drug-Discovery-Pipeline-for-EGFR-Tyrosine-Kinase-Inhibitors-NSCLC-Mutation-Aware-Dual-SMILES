#!/usr/bin/env python3
"""
Prediction script for adv_physchem ChemBERTa Cross-Attention
Hierarchical 6-site RNN Model with ChemBERTa Embeddings

This script loads trained hierarchical site-specific models (with ChemBERTa
cross-attention) and an RNN model to make predictions on new SMILES data.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import pickle
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Concatenate, Multiply,
    LSTM, GRU, Bidirectional, LeakyReLU,
    MultiHeadAttention, LayerNormalization, Add, Reshape, Flatten
)
from tensorflow.keras.optimizers import Adam
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from numpy.linalg import norm
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# === Disk-backed Feature Cache ===
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

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    try:
        # --- inter features (25) ---
        inter = []
        inter.append(Lipinski.NumHDonors(mol))
        inter.append(Lipinski.NumHAcceptors(mol))
        inter.append(Lipinski.NHOHCount(mol))
        inter.append(Lipinski.NOCount(mol))
        inter.append(rdMolDescriptors.CalcNumHBD(mol))
        inter.append(rdMolDescriptors.CalcNumHBA(mol))
        max_pc = Descriptors.MaxPartialCharge(mol)
        min_pc = Descriptors.MinPartialCharge(mol)
        inter.append(max_pc)
        inter.append(min_pc)
        inter.append(Descriptors.MaxAbsPartialCharge(mol))
        inter.append(max_pc - min_pc)
        inter.append(Descriptors.MinAbsPartialCharge(mol))
        inter.append(MolSurf.TPSA(mol))
        inter.append(MolSurf.LabuteASA(mol))
        inter.append(Crippen.MolMR(mol))
        inter.append(Descriptors.MolWt(mol))
        inter.append(Lipinski.HeavyAtomCount(mol))
        inter.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        inter.append(Crippen.MolLogP(mol))
        inter.append(Descriptors.FractionCSP3(mol))
        inter.append(Lipinski.NumAromaticRings(mol))
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        inter.append(aromatic_atoms)
        inter.append(Descriptors.NumAromaticCarbocycles(mol))
        inter.append(Descriptors.NumAromaticHeterocycles(mol))
        inter.append(Fragments.fr_halogen(mol))
        inter.append(Lipinski.NumRotatableBonds(mol))
        inter_arr = np.array(inter)

        # --- intra features (30) ---
        intra = []
        num_bonds = mol.GetNumBonds()
        intra.append(num_bonds)
        single_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 1.0)
        double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 2.0)
        triple_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 3.0)
        aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        intra.extend([single_bonds, double_bonds, triple_bonds, aromatic_bonds])
        avg_bond_order = np.mean([bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]) if num_bonds > 0 else 0
        intra.append(avg_bond_order)
        intra.append(Lipinski.NumRotatableBonds(mol))
        intra.append(Lipinski.RingCount(mol))
        intra.append(Lipinski.NumAromaticRings(mol))
        rigid_bonds = sum(1 for bond in mol.GetBonds() if bond.IsInRing())
        fraction_rigid = rigid_bonds / num_bonds if num_bonds > 0 else 0
        intra.append(fraction_rigid)
        intra.append(Descriptors.NumAromaticCarbocycles(mol))
        intra.append(Descriptors.NumAromaticHeterocycles(mol))
        sp2_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2)
        sp3_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3)
        sp_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP)
        intra.extend([sp_carbons, sp2_carbons, sp3_carbons])
        ring_sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
        avg_ring_size = np.mean(ring_sizes) if ring_sizes else 0
        min_ring_size = min(ring_sizes) if ring_sizes else 0
        intra.extend([avg_ring_size, min_ring_size])
        three_member_rings = sum(1 for size in ring_sizes if size == 3)
        four_member_rings = sum(1 for size in ring_sizes if size == 4)
        intra.extend([three_member_rings, four_member_rings])
        intra.append(GraphDescriptors.BertzCT(mol))
        intra.append(GraphDescriptors.Kappa1(mol))
        intra.append(GraphDescriptors.Kappa2(mol))
        intra.append(GraphDescriptors.Kappa3(mol))
        intra.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
        intra.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        intra_arr = np.array(intra)

        _save_cached_features(smiles, inter_arr, intra_arr)
        return inter_arr, intra_arr

    except Exception:
        return None, None


# ============================================================================
# FEATURE GENERATION FUNCTIONS
# ============================================================================

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value for zero denominator"""
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    else:
        result = np.where(denominator != 0, numerator / denominator, default)
        return result


def generate_lig_inter_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = []

    try:
        features.append(Lipinski.NumHDonors(mol))
        features.append(Lipinski.NumHAcceptors(mol))
        features.append(Lipinski.NHOHCount(mol))
        features.append(Lipinski.NOCount(mol))
        features.append(rdMolDescriptors.CalcNumHBD(mol))
        features.append(rdMolDescriptors.CalcNumHBA(mol))

        features.append(Descriptors.MaxPartialCharge(mol))
        features.append(Descriptors.MinPartialCharge(mol))
        features.append(Descriptors.MaxAbsPartialCharge(mol))
        features.append(Descriptors.MaxPartialCharge(mol) - Descriptors.MinPartialCharge(mol))
        features.append(Descriptors.MinAbsPartialCharge(mol))

        features.append(MolSurf.TPSA(mol))
        features.append(MolSurf.LabuteASA(mol))
        features.append(Crippen.MolMR(mol))

        features.append(Descriptors.MolWt(mol))
        features.append(Lipinski.HeavyAtomCount(mol))
        features.append(rdMolDescriptors.CalcNumRotatableBonds(mol))

        features.append(Crippen.MolLogP(mol))
        features.append(Descriptors.FractionCSP3(mol))
        features.append(Lipinski.NumAromaticRings(mol))
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        features.append(aromatic_atoms)

        features.append(Descriptors.NumAromaticCarbocycles(mol))
        features.append(Descriptors.NumAromaticHeterocycles(mol))

        features.append(Fragments.fr_halogen(mol))

        features.append(Lipinski.NumRotatableBonds(mol))

        return np.array(features)

    except Exception as e:
        print(f"Error in lig_inter: {str(e)}")
        return None


def generate_mut_inter_features(smiles):
    return generate_lig_inter_features(smiles)


def generate_lig_intra_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = []

    try:
        num_bonds = mol.GetNumBonds()
        features.append(num_bonds)

        single_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 1.0)
        double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 2.0)
        triple_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 3.0)
        aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        features.extend([single_bonds, double_bonds, triple_bonds, aromatic_bonds])

        avg_bond_order = np.mean([bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]) if num_bonds > 0 else 0
        features.append(avg_bond_order)

        features.append(Lipinski.NumRotatableBonds(mol))
        features.append(Lipinski.RingCount(mol))
        features.append(Lipinski.NumAromaticRings(mol))

        rigid_bonds = sum(1 for bond in mol.GetBonds() if bond.IsInRing())
        fraction_rigid = rigid_bonds / num_bonds if num_bonds > 0 else 0
        features.append(fraction_rigid)

        features.append(Descriptors.NumAromaticCarbocycles(mol))
        features.append(Descriptors.NumAromaticHeterocycles(mol))

        sp2_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2)
        sp3_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3)
        sp_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP)
        features.extend([sp_carbons, sp2_carbons, sp3_carbons])

        ring_sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
        avg_ring_size = np.mean(ring_sizes) if ring_sizes else 0
        min_ring_size = min(ring_sizes) if ring_sizes else 0
        features.extend([avg_ring_size, min_ring_size])

        three_member_rings = sum(1 for size in ring_sizes if size == 3)
        four_member_rings = sum(1 for size in ring_sizes if size == 4)
        features.extend([three_member_rings, four_member_rings])

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
    norm1 = norm(vec1)
    norm2 = norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return {
            'cosine_similarity': 0.0,
            'sine_dissimilarity': 0.0,
            'dot_product': 0.0
        }

    cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    sine_of_angle = np.sqrt(1 - cosine_sim**2)

    return {
        'cosine_similarity': cosine_sim,
        'sine_dissimilarity': sine_of_angle,
        'dot_product': np.dot(vec1, vec2)
    }


def calculate_fp_metrics(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return {'dice_sim': 0.0, 'tanimato': 0.0}

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

    dice_sim = DataStructs.DiceSimilarity(fp1, fp2)
    tanimato = DataStructs.TanimotoSimilarity(fp1, fp2)

    return {
        'dice_sim': dice_sim,
        'tanimato': tanimato,
    }


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
    """Generate custom intermolecular and intramolecular features with safe division"""
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


def generate_all_features(lig_smiles, mut_smiles):
    """
    Generate all feature sets for a given ligand-mutation pair.
    Returns a dictionary matching the keys used in prediction loop.
    """
    lig_inter = generate_lig_inter_features(lig_smiles)
    lig_intra = generate_lig_intra_features(lig_smiles)
    mut_inter = generate_mut_inter_features(mut_smiles)
    mut_intra = generate_mut_intra_features(mut_smiles)

    if any(x is None for x in [lig_inter, lig_intra, mut_inter, mut_intra]):
        return None

    lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra = generate_custom_features(
        lig_inter, mut_inter, lig_intra, mut_intra
    )

    inter_interaction = generate_inter_interaction_features(lig_inter, mut_inter)
    intra_interaction = generate_intra_interaction_features(lig_intra, mut_intra)

    if len(lig_mut_inter) > 0:
        inter_interaction = np.concatenate([np.array(lig_mut_inter), inter_interaction])

    if len(lig_mut_intra) > 0:
        intra_interaction = np.concatenate([np.array(lig_mut_intra), intra_interaction])

    final_fp_interaction = generate_final_interaction_features(lig_smiles, mut_smiles)

    return {
        'lig_inter': lig_inter,
        'mut_inter': mut_inter,
        'inter_interaction': inter_interaction,
        'lig_intra': lig_intra,
        'mut_intra': mut_intra,
        'intra_interaction': intra_interaction,
        'lig_mut_mix_inter_intra': np.array(lig_mut_mix_inter_intra),
        'final_fp_interaction': final_fp_interaction
    }


# ============================================================================
# CHEMBERTA FUNCTIONS
# ============================================================================

def get_device():
    return torch.device('cpu')


def load_chemberta(model_name='seyonec/ChemBERTa-zinc-base-v1', device=None):
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


# ============================================================================
# BANNER
# ============================================================================

print("=" * 80)
print("PREDICTION SCRIPT FOR ADV_PHYSCHEM CHEMBERTA CROSS-ATTENTION")
print("Hierarchical 6-site RNN Model with ChemBERTa Embeddings")
print("=" * 80)

MUTATION_SITES = ['FULL_SMILES', 'ATP_POCKET', 'P_LOOP_HINGE', 'C_HELIX', 'DFG_A_LOOP', 'HRD_CAT']
SITE_COLUMNS = ['smiles_full_egfr', 'smiles 718_862_atp_pocket', 'smiles_p_loop', 'smiles_c_helix', 'smiles_l858r_a_loop_dfg_motif', 'smiles_catalytic_hrd_motif']


def load_models_and_scalers(model_dir):
    """Load all required models, scalers, and ChemBERTa scaler"""

    print("\nLoading models and scalers...")

    # Load hierarchical models for each site
    hierarchical_models = {}
    for site_name in MUTATION_SITES:
        model_path = os.path.join(model_dir, f'hierarchical_model_{site_name}.h5')
        if os.path.exists(model_path):
            hierarchical_models[site_name] = load_model(model_path, compile=False)
            print(f"  Loaded {site_name} model")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    # Load RNN model
    rnn_path = os.path.join(model_dir, 'rnn_sequential_model.h5')
    if os.path.exists(rnn_path):
        rnn_model = load_model(rnn_path, compile=False)
        print(f"  Loaded RNN model")
    else:
        raise FileNotFoundError(f"RNN model not found: {rnn_path}")

    # Load feature scalers
    with open(os.path.join(model_dir, 'feature_scalers.pkl'), 'rb') as f:
        all_scalers = pickle.load(f)
    print(f"  Loaded feature scalers")

    # Load y scalers
    with open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb') as f:
        y_scalers = pickle.load(f)
    y_scaler1 = y_scalers['y_scaler1']
    y_scaler2 = y_scalers['y_scaler2']
    print(f"  Loaded y scalers")

    # Load ChemBERTa scalers
    with open(os.path.join(model_dir, 'chembert_scalers.pkl'), 'rb') as f:
        chem_emb_scaler = pickle.load(f)[0]
    print(f"  Loaded ChemBERTa embedding scaler")

    return hierarchical_models, rnn_model, all_scalers, y_scaler1, y_scaler2, chem_emb_scaler


# ============================================================================
# EVALUATION AND PLOTTING FUNCTION
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
    print(f"\nMetrics saved to: {metrics_csv_path}")

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
    print(f"Overall combined plot saved to: {overall_plot_file}")
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
        print(f"  Saved: {mutation_safe}.png")
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
        print(f"  Saved: pearson_{mutation_safe}.png")
        plt.close()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Metrics CSV: {metrics_csv_path}")
    print(f"Overall plot: {overall_plot_file}")
    print(f"Individual mutation plots: {plots_dir}")
    print(f"Pearson correlation plots: {pearson_dir}")
    print("=" * 80)


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_predictions(input_csv, model_dir='.', output_dir='.'):
    """
    Make predictions using the trained hierarchical RNN model with ChemBERTa
    cross-attention. Optimized: batch processing grouped by mutation, embedding
    models created once.

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
    df_pred = pd.read_csv(input_csv, encoding='latin-1')

    # Check required columns
    required_cols = ['smiles', 'tkd']
    if not all(col in df_pred.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain 'smiles' and 'tkd' columns. Found: {df_pred.columns.tolist()}")

    # Check if ground truth exists
    has_ground_truth = 'standard value' in df_pred.columns and 'dock' in df_pred.columns

    # Load Model Files and Scalers
    print("Loading model files...")
    try:
        hierarchical_models, rnn_model, all_scalers, y_scaler1, y_scaler2, chem_emb_scaler = load_models_and_scalers(model_dir)

        # Load mutation profiles
        mutation_profiles_path = os.path.join(model_dir, 'mutation_profiles.csv')
        if not os.path.exists(mutation_profiles_path):
            raise FileNotFoundError(f"Mutation profiles not found: {mutation_profiles_path}")

        df_mutation_profiles = pd.read_csv(mutation_profiles_path)
        print(f"  Loaded {len(df_mutation_profiles)} mutation profiles")

    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please ensure all model files are in the model directory:")
        print("  - hierarchical_model_*.h5 (6 files)")
        print("  - rnn_sequential_model.h5")
        print("  - feature_scalers.pkl")
        print("  - y_scalers.pkl")
        print("  - chembert_scalers.pkl")
        print("  - mutation_profiles.csv")
        return None

    print(f"\nTotal prediction samples: {len(df_pred)}")

    # Load ChemBERTa model
    print("\nLoading ChemBERTa model...")
    tokenizer, chem_model, device = load_chemberta()
    print(f"  ChemBERTa loaded on device: {device}")

    # Recompile hierarchical models
    print("\nRecompiling hierarchical models...")
    for site_name, model in hierarchical_models.items():
        model.compile(
            optimizer=Adam(learning_rate=0.003),
            loss={
                'activity_output': 'mean_squared_error',
                'docking_output': 'mean_squared_error'
            },
            loss_weights={
                'activity_output': 1.0,
                'docking_output': 0.7
            },
            metrics={
                'activity_output': ['mae', 'mse'],
                'docking_output': ['mae', 'mse']
            }
        )

    # Recompile RNN model
    rnn_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'final_activity_output': 'mean_squared_error',
            'final_docking_output': 'mean_squared_error'
        },
        loss_weights={
            'final_activity_output': 1.0,
            'final_docking_output': 0.5
        },
        metrics={
            'final_activity_output': ['mae', 'mse'],
            'final_docking_output': ['mae', 'mse']
        }
    )

    # Pre-create embedding models once (instead of inside per-sample loop)
    embedding_models = {}
    for site_name in MUTATION_SITES:
        h_model = hierarchical_models[site_name]
        embedding_models[site_name] = Model(
            inputs=h_model.inputs,
            outputs=h_model.get_layer('embedding_output').output
        )
    print(f"Pre-created {len(embedding_models)} embedding models")

    # Site ordering for the prediction loop (must match training script order)
    site_order = [
        ('ATP_POCKET', 'smiles 718_862_atp_pocket'),
        ('P_LOOP_HINGE', 'smiles_p_loop'),
        ('C_HELIX', 'smiles_c_helix'),
        ('DFG_A_LOOP', 'smiles_l858r_a_loop_dfg_motif'),
        ('HRD_CAT', 'smiles_catalytic_hrd_motif'),
        ('FULL_SMILES', 'smiles_full_egfr')
    ]

    # Batch processing: group by mutation type
    results = []
    unique_mutations = df_pred['tkd'].unique()
    print(f"\nProcessing {len(unique_mutations)} unique mutations in batch mode...")

    # In-memory ligand feature cache for this run
    _lig_feature_cache = {}

    for mutation_name in unique_mutations:
        mut_data = df_pred[df_pred['tkd'] == mutation_name]

        # Find mutation profile
        mutation_profile = df_mutation_profiles[df_mutation_profiles['tkd'] == mutation_name]
        if len(mutation_profile) == 0:
            print(f"  Warning: Mutation '{mutation_name}' not found in training data, skipping {len(mut_data)} samples")
            continue
        mutation_profile = mutation_profile.iloc[0]

        # Get mutation site SMILES
        mut_site_smiles = {}
        skip_mutation = False
        for site_name, site_col in site_order:
            smi = mutation_profile[site_col]
            if pd.isna(smi) or smi == '':
                print(f"  Warning: Missing SMILES for {mutation_name} at {site_name}, skipping")
                skip_mutation = True
                break
            mut_site_smiles[site_name] = smi
        if skip_mutation:
            continue

        print(f"\n  Processing mutation: {mutation_name} ({len(mut_data)} compounds)")

        # Pre-compute mutation features for each site (shared across all ligands)
        mut_features_by_site = {}
        skip_mutation = False
        for site_name, _ in site_order:
            mut_smi = mut_site_smiles[site_name]
            mut_inter, mut_intra = _generate_lig_features(mut_smi)
            if mut_inter is None or mut_intra is None:
                print(f"    Warning: Failed mutation features for {site_name}")
                skip_mutation = True
                break
            mut_features_by_site[site_name] = (mut_inter, mut_intra)
        if skip_mutation:
            continue

        # Batch-compute ligand features for all compounds in this mutation group
        valid_rows = []  # (original_idx, row, lig_inter, lig_intra)
        for idx, row in mut_data.iterrows():
            lig_smiles = row['smiles']
            if pd.isna(lig_smiles) or lig_smiles == '':
                continue

            if lig_smiles in _lig_feature_cache:
                lig_inter, lig_intra = _lig_feature_cache[lig_smiles]
            else:
                lig_inter, lig_intra = _generate_lig_features(lig_smiles)
                if lig_inter is not None and lig_intra is not None:
                    _lig_feature_cache[lig_smiles] = (lig_inter, lig_intra)

            if lig_inter is None or lig_intra is None:
                continue
            valid_rows.append((idx, row, lig_inter, lig_intra))

        if not valid_rows:
            print(f"    No valid compounds for {mutation_name}")
            continue

        n_valid = len(valid_rows)

        # For each site, batch-generate all features and batch-predict embeddings
        embeddings_all_sites = []  # list of (n_valid, embedding_dim) arrays
        site_failed = False

        for site_idx, (site_name, _) in enumerate(site_order):
            mut_inter, mut_intra = mut_features_by_site[site_name]
            mut_smi = mut_site_smiles[site_name]

            # Batch-build feature arrays
            batch_features = {
                'lig_inter': [], 'mut_inter': [], 'inter_interaction': [],
                'lig_intra': [], 'mut_intra': [], 'intra_interaction': [],
                'lig_mut_mix_inter_intra': [], 'final_fp_interaction': []
            }

            batch_valid = []
            ligand_smiles_list = []
            for i, (orig_idx, row, lig_inter, lig_intra) in enumerate(valid_rows):
                lig_smiles = row['smiles']

                lig_mut_inter, lig_mut_intra, lig_mut_mix = generate_custom_features(
                    lig_inter, mut_inter, lig_intra, mut_intra
                )
                inter_interaction = generate_inter_interaction_features(lig_inter, mut_inter)
                intra_interaction = generate_intra_interaction_features(lig_intra, mut_intra)

                if len(lig_mut_inter) > 0:
                    inter_interaction = np.concatenate([np.array(lig_mut_inter), inter_interaction])
                if len(lig_mut_intra) > 0:
                    intra_interaction = np.concatenate([np.array(lig_mut_intra), intra_interaction])

                final_fp = generate_final_interaction_features(lig_smiles, mut_smi)

                batch_features['lig_inter'].append(lig_inter)
                batch_features['mut_inter'].append(mut_inter)
                batch_features['inter_interaction'].append(inter_interaction)
                batch_features['lig_intra'].append(lig_intra)
                batch_features['mut_intra'].append(mut_intra)
                batch_features['intra_interaction'].append(intra_interaction)
                batch_features['lig_mut_mix_inter_intra'].append(np.array(lig_mut_mix))
                batch_features['final_fp_interaction'].append(final_fp)
                batch_valid.append(i)
                ligand_smiles_list.append(lig_smiles)

            if not batch_valid:
                site_failed = True
                break

            # Batch scale standard features
            scalers = all_scalers[site_idx]
            scaled = {}
            for key in batch_features:
                scaled[key] = scalers[key].transform(np.array(batch_features[key]))

            # Compute ChemBERTa embeddings for ligands
            lig_embs = get_chemberta_embeddings(ligand_smiles_list, tokenizer, chem_model, device)
            lig_embs = chem_emb_scaler.transform(lig_embs)

            # Compute ChemBERTa embeddings for mutation site
            mut_embs = get_chemberta_embeddings([mut_smi] * n_valid, tokenizer, chem_model, device)
            mut_embs = chem_emb_scaler.transform(mut_embs)

            # Batch predict embeddings (10 inputs: 2 ChemBERTa + 8 standard)
            site_emb = embedding_models[site_name].predict([
                lig_embs,
                mut_embs,
                scaled['final_fp_interaction'],
                scaled['lig_mut_mix_inter_intra'],
                scaled['inter_interaction'],
                scaled['intra_interaction'],
                scaled['mut_inter'],
                scaled['lig_inter'],
                scaled['mut_intra'],
                scaled['lig_intra']
            ], verbose=0)

            embeddings_all_sites.append(site_emb)

        if site_failed or len(embeddings_all_sites) != 6:
            print(f"    Failed to process all sites for {mutation_name}")
            continue

        # Stack embeddings: (n_valid, 6, embedding_dim)
        sequential_input = np.stack(embeddings_all_sites, axis=1)

        # Batch predict through RNN
        predictions = rnn_model.predict(sequential_input, verbose=0)
        pred_activity_scaled = predictions[0].flatten()
        pred_docking_scaled = predictions[1].flatten()

        # Batch inverse transform
        pred_activity_log1p = y_scaler1.inverse_transform(pred_activity_scaled.reshape(-1, 1)).flatten()
        pred_activity = np.expm1(pred_activity_log1p)
        pred_docking = y_scaler2.inverse_transform(pred_docking_scaled.reshape(-1, 1)).flatten()

        # Store results
        for i, (orig_idx, row, _, _) in enumerate(valid_rows):
            res = {
                'smiles': row['smiles'],
                'mutation': mutation_name,
                'predicted_activity': pred_activity[i],
                'predicted_docking': pred_docking[i]
            }

            if has_ground_truth:
                res['actual_activity'] = row['standard value']
                res['actual_docking'] = row['dock']

            for col in df_pred.columns:
                if col not in res and col not in ['smiles', 'standard value', 'dock']:
                    res[col] = row[col]

            results.append(res)

        print(f"    Predicted {n_valid} compounds for {mutation_name}")

    if not results:
        print("Error: No valid samples found to predict.")
        return None

    df_results = pd.DataFrame(results)

    # Save output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'predictions_chembert_crossattention.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    print(f"\n--- Feature Cache Stats ---")
    print(f"  Disk cache hits:   {_cache_hits}")
    print(f"  Disk cache misses: {_cache_misses}")
    print(f"  Cache directory:   {os.path.abspath(_CACHE_DIR)}")

    # Evaluation (if ground truth exists)
    if has_ground_truth and len(df_results) > 0:
        evaluate_and_plot(df_results, output_dir, 'chembert_crossattention')

    return df_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predictions for adv_physchem ChemBERTa Cross-Attention')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--model_dir', type=str, default='.', help='Model directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: {model_dir}/predictions/)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'predictions')

    results = make_predictions(args.input, args.model_dir, args.output_dir)

    print(f"\nComplete! Total predictions: {len(results)}")
    print(f"Mutations covered: {results['mutation'].nunique()}")
