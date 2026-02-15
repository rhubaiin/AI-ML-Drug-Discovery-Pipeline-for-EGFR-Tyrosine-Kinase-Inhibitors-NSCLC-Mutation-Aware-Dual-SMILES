#!/usr/bin/env python3
"""
Prediction script for adv_physchem_gnn_base1a.py
GNN (MolCLR) with Graph Isomorphism Network

This script loads trained GNN hierarchical models and makes predictions on new SMILES data.

Requirements:
    pip install torch torch-geometric torch-scatter torch-sparse
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# PyTorch imports for MolCLR
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import torch_geometric - required for full GNN functionality
TORCH_GEOMETRIC_AVAILABLE = False
try:
    from torch_geometric.nn import MessagePassing, global_mean_pool
    from torch_geometric.utils import add_self_loops
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("\n" + "!"*80)
    print("WARNING: torch_geometric not installed!")
    print("To install: pip install torch-geometric torch-scatter torch-sparse")
    print("Running in FALLBACK mode with simple MLP embeddings")
    print("!"*80 + "\n")
    # Fallback placeholders
    MessagePassing = nn.Module
    global_mean_pool = None
    add_self_loops = None
    Data = None
    Batch = None

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

NUM_ATOM_TYPE = 119  # including mask tokens
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 5    # including aromatic and self-loop
NUM_BOND_DIRECTION = 3


if TORCH_GEOMETRIC_AVAILABLE:
    class GINEConv(MessagePassing):
        """Graph Isomorphism Network with Edge features."""
        def __init__(self, emb_dim):
            super(GINEConv, self).__init__(aggr='add')
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, 2*emb_dim),
                nn.ReLU(),
                nn.Linear(2*emb_dim, emb_dim)
            )
            self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
            self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
            nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
            nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        def forward(self, x, edge_index, edge_attr):
            edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
            self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
            self_loop_attr[:, 0] = 4
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
            edge_embeddings = self.edge_embedding1(edge_attr[:, 0].long()) + \
                             self.edge_embedding2(edge_attr[:, 1].long())
            return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

        def message(self, x_j, edge_attr):
            return x_j + edge_attr

        def update(self, aggr_out):
            return self.mlp(aggr_out)

    class GINet(nn.Module):
        """MolCLR Graph Isomorphism Network for molecular embeddings."""
        def __init__(self, num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.0, pool='mean'):
            super(GINet, self).__init__()
            self.num_layer = num_layer
            self.emb_dim = emb_dim
            self.feat_dim = feat_dim
            self.drop_ratio = drop_ratio
            self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
            self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
            nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            nn.init.xavier_uniform_(self.x_embedding2.weight.data)
            self.gnns = nn.ModuleList([GINEConv(emb_dim) for _ in range(num_layer)])
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])
            self.pool = global_mean_pool
            self.feat_lin = nn.Linear(emb_dim, feat_dim)

        def forward(self, data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
            for layer in range(self.num_layer):
                h = self.gnns[layer](h, edge_index, edge_attr)
                h = self.batch_norms[layer](h)
                if layer == self.num_layer - 1:
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h = self.pool(h, batch)
            h = self.feat_lin(h)
            return h

        def load_pretrained(self, state_dict):
            own_state = self.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                if isinstance(param, nn.parameter.Parameter):
                    param = param.data
                own_state[name].copy_(param)

else:
    # Fallback: Simple MLP-based molecular embedding (no graph structure)
    class GINet(nn.Module):
        """Fallback MLP embedder when torch_geometric not available."""
        def __init__(self, num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.0, pool='mean'):
            super(GINet, self).__init__()
            self.feat_dim = feat_dim
            # Simple MLP that produces fixed-size embeddings from molecular fingerprints
            self.mlp = nn.Sequential(
                nn.Linear(2048, 512),  # Morgan fingerprint size
                nn.ReLU(),
                nn.Dropout(drop_ratio),
                nn.Linear(512, feat_dim),
                nn.ReLU()
            )
            
        def forward(self, fingerprints):
            """Accept fingerprint tensor directly instead of graph data."""
            return self.mlp(fingerprints)
        
        def load_pretrained(self, state_dict):
            pass  # No pretrained weights for fallback


def smiles_to_graph(smiles):
    """Convert SMILES string to PyTorch Geometric Data object."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None  # Use fingerprint-based approach in fallback mode
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features: [atom_type, chirality]
    atom_features = []
    for atom in mol.GetAtoms():
        atom_type = atom.GetAtomicNum()
        # Cap atomic number at 118
        if atom_type > 118:
            atom_type = 118
            
        chirality = int(atom.GetChiralTag())
        # Map chirality to 0-2 range (0=unspecified, 1=CW, 2=CCW)
        # 3 (CHI_OTHER) and others map to 0
        if chirality > 2:
            chirality = 0
            
        atom_features.append([atom_type, chirality])
    
    x = torch.tensor(atom_features, dtype=torch.long)
    
    # Edge features: [bond_type, bond_direction]
    edge_index = []
    edge_attr = []
    
    # Bond type mapping: Single->0, Double->1, Triple->2, Aromatic->3
    bond_mapping = {
        1: 0,  # SINGLE
        2: 1,  # DOUBLE
        3: 2,  # TRIPLE
        12: 3, # AROMATIC
    }
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Bond Type
        bt_val = int(bond.GetBondType())
        bond_type = bond_mapping.get(bt_val, 0) # Default to SINGLE (0) if unknown
        
        # Bond Direction
        bond_dir = int(bond.GetBondDir())
        # Map RDKit BondDir to 0-2 range
        # NONE=0, BEGINWEDGE=1, BEGINDASH=2, ENDDOWN=3, ENDUP=4, EITHERDOUBLE=5, UNKNOWN=6
        if bond_dir > 2:
            bond_dir = 0
            
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([[bond_type, bond_dir], [bond_type, bond_dir]])
    
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_fingerprint(smiles, nBits=2048):
    """Convert SMILES to Morgan fingerprint for fallback mode."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    arr = np.zeros(nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_gnn_embeddings(smiles_list, model, device, batch_size=32):
    """Extract GNN embeddings for a list of SMILES strings."""

    logger.debug("="*80)
    logger.debug("FUNCTION CALL: get_gnn_embeddings")
    logger.debug(f"Input parameters:")
    logger.debug(f"  - smiles_list length: {len(smiles_list)}")
    logger.debug(f"  - device: {device}")
    logger.debug(f"  - batch_size: {batch_size}")
    logger.debug(f"  - torch_geometric available: {TORCH_GEOMETRIC_AVAILABLE}")
    logger.debug(f"  - model.feat_dim: {model.feat_dim}")
    logger.debug(f"  - model.training: {model.training}")
    logger.debug(f"First 3 SMILES: {smiles_list[:3]}")

    model.eval()
    all_embeddings = []
    
    if TORCH_GEOMETRIC_AVAILABLE:
        # Full GNN mode with graph data
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            graphs = []
            valid_indices = []
            
            for idx, smi in enumerate(batch_smiles):
                graph = smiles_to_graph(smi)
                if graph is not None:
                    graphs.append(graph)
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Failed to convert SMILES to graph: {smi}")

            logger.debug(f"Batch stats: {len(graphs)} valid graphs / {len(batch_smiles)} total")
        
            if len(graphs) == 0:
                all_embeddings.extend([np.zeros(model.feat_dim) for _ in batch_smiles])
                continue
            
            batch = Batch.from_data_list(graphs).to(device)

            logger.debug(f"Batch object created:")
            logger.debug(f"  - num_nodes: {batch.num_nodes}")
            logger.debug(f"  - num_edges: {batch.num_edges}")
            logger.debug(f"  - num_graphs: {batch.num_graphs}")
            logger.debug(f"  - device: {batch.x.device}")
            
            with torch.no_grad():
                embeddings = model(batch).cpu().numpy()

            logger.debug(f"Embeddings generated:")
            logger.debug(f"  - shape: {embeddings.shape}")
            logger.debug(f"  - dtype: {embeddings.dtype}")
            logger.debug(f"  - range: [{embeddings.min():.6f}, {embeddings.max():.6f}]")
            logger.debug(f"  - mean: {embeddings.mean():.6f}, std: {embeddings.std():.6f}")
            logger.debug(f"  - NaN count: {np.isnan(embeddings).sum()}")
            logger.debug(f"  - Inf count: {np.isinf(embeddings).sum()}")
            
            batch_result = [np.zeros(model.feat_dim) for _ in batch_smiles]
            for j, orig_idx in enumerate(valid_indices):
                batch_result[orig_idx] = embeddings[j]
            all_embeddings.extend(batch_result)
    else:
        logger.debug("Unable to get_gnn_embeddings")
        sys.exit(1)
 

    final_array = np.array(all_embeddings)
    logger.success(f"get_gnn_embeddings COMPLETE:")
    logger.success(f"  - Output shape: {final_array.shape}")
    logger.success(f"  - Expected: ({len(smiles_list)}, {model.feat_dim})")
    logger.success(f"  - Final range: [{final_array.min():.6f}, {final_array.max():.6f}]")
    logger.success(f"  - Final mean: {final_array.mean():.6f}, std: {final_array.std():.6f}")
    logger.debug("="*80)
    
    return np.array(all_embeddings)


# =============================================================================
# Feature Generation Functions (from adv_physchem5f2.py)
# =============================================================================

def safe_divide(numerator, denominator, default=0.0):
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    return np.where(denominator != 0, numerator / denominator, default)


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
        logger.warning(f"Error in lig_inter: {e}")
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
        sp2 = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2)
        sp3 = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP3)
        sp = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP)
        features.extend([sp, sp2, sp3])
        ring_sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
        avg_ring_size = np.mean(ring_sizes) if ring_sizes else 0
        min_ring_size = min(ring_sizes) if ring_sizes else 0
        features.extend([avg_ring_size, min_ring_size])
        three_member = sum(1 for s in ring_sizes if s == 3)
        four_member = sum(1 for s in ring_sizes if s == 4)
        features.extend([three_member, four_member])
        features.append(GraphDescriptors.BertzCT(mol))
        features.append(GraphDescriptors.Kappa1(mol))
        features.append(GraphDescriptors.Kappa2(mol))
        features.append(GraphDescriptors.Kappa3(mol))
        features.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
        features.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        return np.array(features)
    except Exception as e:
        logger.warning(f"Error in lig_intra: {e}")
        return None


def generate_mut_intra_features(smiles):
    return generate_lig_intra_features(smiles)


def calculate_similarity_metrics(vec1, vec2):
    norm1, norm2 = norm(vec1), norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return {'cosine_similarity': 0.0, 'sine_dissimilarity': 0.0, 'dot_product': 0.0}
    cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    sine_of_angle = np.sqrt(max(0, 1 - cosine_sim**2))
    return {'cosine_similarity': cosine_sim, 'sine_dissimilarity': sine_of_angle, 
            'dot_product': np.dot(vec1, vec2)}


def calculate_fp_metrics(smiles1, smiles2):
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return {'dice_sim': 0.0, 'tanimato': 0.0}
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return {'dice_sim': DataStructs.DiceSimilarity(fp1, fp2), 
            'tanimato': DataStructs.TanimotoSimilarity(fp1, fp2)}


def generate_inter_interaction_features(lig_inter, mut_inter):
    metrics = calculate_similarity_metrics(lig_inter, mut_inter)
    return np.array([metrics['cosine_similarity'], metrics['sine_dissimilarity']])


def generate_intra_interaction_features(lig_intra, mut_intra):
    metrics = calculate_similarity_metrics(lig_intra, mut_intra)
    return np.array([metrics['cosine_similarity'], metrics['sine_dissimilarity']])


def generate_final_interaction_features(lig_smiles, mut_smiles):
    fp_metrics = calculate_fp_metrics(lig_smiles, mut_smiles)
    return np.array([fp_metrics['dice_sim'], fp_metrics['tanimato']])


def generate_custom_features(lig_inter, mut_inter, lig_intra, mut_intra):
    lig_mut_inter = []
    lig_mut_intra = []
    lig_mut_mix = []
    
    # H-bonding features
    lig_mut_inter.append(safe_divide(lig_inter[0] * mut_inter[1], mut_inter[0]))
    lig_mut_inter.append(safe_divide(lig_inter[4] * mut_inter[5], mut_inter[4]))
    lig_mut_mix.append(safe_divide(safe_divide(lig_inter[0]*mut_inter[1], mut_inter[0]), mut_intra[21]))
    lig_mut_inter.append(safe_divide(lig_inter[0]*mut_inter[1], lig_inter[1]) + 
                         safe_divide(lig_inter[1]*mut_inter[0], mut_inter[1]))
    lig_mut_inter.append(safe_divide(lig_inter[4]*mut_inter[5], lig_inter[4]) + 
                         safe_divide(lig_inter[5]*mut_inter[4], mut_inter[5]))
    lig_mut_inter.append(safe_divide(lig_inter[0], lig_inter[1]) + 
                         safe_divide(mut_inter[1], mut_inter[0]))
    lig_mut_inter.append(safe_divide(lig_inter[4], lig_inter[5]) + 
                         safe_divide(mut_inter[5], mut_inter[4]))
    
    # Electrostatic features
    lig_mut_mix.append(safe_divide(lig_inter[6], lig_intra[14]) * safe_divide(mut_inter[7], mut_intra[14]))
    lig_mut_mix.append(safe_divide(lig_inter[7], lig_intra[14]) * safe_divide(mut_inter[6], mut_intra[14]))
    c_l1 = safe_divide(lig_inter[6], lig_intra[14]) * safe_divide(mut_inter[7], mut_intra[14])
    c_l2 = safe_divide(lig_inter[7], lig_intra[14]) * safe_divide(mut_inter[6], mut_intra[14])
    lig_mut_mix.append(c_l1**2 + c_l2**2)
    
    lig_mut_inter.append((lig_inter[6] - mut_inter[7]) - (mut_inter[6] - lig_inter[7]))
    lig_mut_inter.append(lig_inter[11] - mut_inter[11])
    lig_mut_inter.append(lig_inter[17] - mut_inter[17])
    lig_mut_inter.append(safe_divide(lig_inter[11]*mut_inter[11], lig_inter[17]*mut_inter[17]))
    
    # Pi-pi stacking
    lig_mut_mix.append(safe_divide(lig_inter[21]+lig_inter[22]+mut_inter[21]+mut_inter[22], 
                                   lig_intra[15]+mut_intra[15]))
    lig_mut_mix.append(safe_divide(lig_inter[21]+lig_inter[22]+mut_inter[21]+mut_inter[22], 
                                   lig_intra[22]+mut_intra[22]))
    
    # Bond rigidity
    bond_rigid = (safe_divide(lig_intra[2]+lig_intra[3]+lig_intra[4], lig_intra[0]) + 
                  safe_divide(mut_intra[2]+mut_intra[3]+mut_intra[4], mut_intra[0]))
    bond_single = safe_divide(lig_intra[1], lig_intra[0]) + safe_divide(mut_intra[1], mut_intra[0])
    lig_mut_intra.append((bond_single - bond_rigid)**2)
    
    # Hybridization
    hyb_lig = safe_divide(lig_intra[12]+lig_intra[13], lig_intra[14]+lig_intra[12]+lig_intra[13])
    hyb_mut = safe_divide(mut_intra[12]+mut_intra[13], mut_intra[14]+mut_intra[12]+mut_intra[13])
    lig_mut_intra.append((hyb_mut - hyb_lig)**2)
    lig_mut_intra.append(safe_divide(lig_intra[21], mut_intra[21]))
    
    return lig_mut_inter, lig_mut_intra, lig_mut_mix


def generate_hierarchical_features(ligand_smiles_series, mutation_smiles_series):
    print('\nGenerating hierarchical features...')
    ligand_cache, mutation_cache, interaction_cache = {}, {}, {}
    
    results = {k: [] for k in ['lig_inter', 'mut_inter', 'inter_interaction', 'lig_intra', 
                                'mut_intra', 'intra_interaction', 'lig_mut_mix_inter_intra', 
                                'final_fp_interaction']}
    valid_indices = []
    
    for idx, (lig_smi, mut_smi) in enumerate(zip(ligand_smiles_series, mutation_smiles_series)):
        if idx % 50 == 0:
            print(f'  Processing sample {idx}/{len(ligand_smiles_series)}...')
        
        # Cache ligand features
        if lig_smi in ligand_cache:
            lig_inter, lig_intra = ligand_cache[lig_smi]
        else:
            lig_inter = generate_lig_inter_features(lig_smi)
            lig_intra = generate_lig_intra_features(lig_smi)
            ligand_cache[lig_smi] = (lig_inter, lig_intra)
        
        # Cache mutation features
        if mut_smi in mutation_cache:
            mut_inter, mut_intra = mutation_cache[mut_smi]
        else:
            mut_inter = generate_mut_inter_features(mut_smi)
            mut_intra = generate_mut_intra_features(mut_smi)
            mutation_cache[mut_smi] = (mut_inter, mut_intra)
        
        if any(f is None for f in [lig_inter, mut_inter, lig_intra, mut_intra]):
            continue
        
        # Cache interaction features
        pair_key = (lig_smi, mut_smi)
        if pair_key in interaction_cache:
            lig_mut_mix, inter_int, intra_int, fp_int = interaction_cache[pair_key]
        else:
            lig_mut_inter, lig_mut_intra, lig_mut_mix = generate_custom_features(
                lig_inter, mut_inter, lig_intra, mut_intra)
            inter_int = generate_inter_interaction_features(lig_inter, mut_inter)
            intra_int = generate_intra_interaction_features(lig_intra, mut_intra)
            if lig_mut_inter:
                inter_int = np.concatenate([np.array(lig_mut_inter), inter_int])
            if lig_mut_intra:
                intra_int = np.concatenate([np.array(lig_mut_intra), intra_int])
            fp_int = generate_final_interaction_features(lig_smi, mut_smi)
            interaction_cache[pair_key] = (lig_mut_mix, inter_int, intra_int, fp_int)
        
        results['lig_inter'].append(lig_inter)
        results['mut_inter'].append(mut_inter)
        results['inter_interaction'].append(inter_int)
        results['lig_intra'].append(lig_intra)
        results['mut_intra'].append(mut_intra)
        results['intra_interaction'].append(intra_int)
        results['lig_mut_mix_inter_intra'].append(np.array(lig_mut_mix))
        results['final_fp_interaction'].append(fp_int)
        valid_indices.append(idx)
    
    print(f'  Successfully generated features for {len(valid_indices)} samples')
    result_dict = {k: np.array(v) for k, v in results.items()}
    result_dict['valid_indices'] = valid_indices
    return result_dict



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
        model_path = os.path.join(model_dir, f'gnn_hierarchical_{site_name}.h5')
        if os.path.exists(model_path):
            hierarchical_models[site_name] = load_model(model_path, compile=False)
            print(f"  ✓ Loaded {site_name} model")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load RNN model
    rnn_path = os.path.join(model_dir, 'gnn_rnn_model.h5')
    if os.path.exists(rnn_path):
        rnn_model = load_model(rnn_path, compile=False)
        print(f"  ✓ Loaded RNN model")
    else:
        raise FileNotFoundError(f"RNN model not found: {rnn_path}")
    
    # Load feature scalers
    with open(os.path.join(model_dir, 'gnn_feature_scalers.pkl'), 'rb') as f:
        all_scalers = pickle.load(f)
    print(f"  ✓ Loaded feature scalers")
    
    # Load y scalers
    with open(os.path.join(model_dir, 'gnn_y_scalers.pkl'), 'rb') as f:
        y_scalers = pickle.load(f)
    y_scaler1 = y_scalers['y_scaler1']
    y_scaler2 = y_scalers['y_scaler2']
    print(f"  ✓ Loaded y scalers")
    
    # Load GNN embedding scaler
    with open(os.path.join(model_dir, 'gnn_embedding_scalers.pkl'), 'rb') as f:
        gnn_emb_scaler = pickle.load(f)
    print(f"  ✓ Loaded GNN embedding scaler")
    
    # Initialize GNN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.0).to(device)
    gnn_model.eval()
    print(f"  ✓ Initialized GNN model on {device}")
    
    # Try to load pretrained weights if available
    pretrained_path = os.path.join(model_dir, 'gnn_pretrained.pt')
    if os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device)
        gnn_model.load_pretrained(state_dict)
        print(f"  ✓ Loaded pretrained GNN weights")
    else:
        print(f"  ℹ Using random GNN initialization (no pretrained weights found)")
    
    return hierarchical_models, rnn_model, all_scalers, y_scaler1, y_scaler2, gnn_emb_scaler, gnn_model, device

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
    
    parser = argparse.ArgumentParser(description='Predictions for GNN (MolCLR) model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--model_dir', type=str, default='.', help='Model directory')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    results = make_predictions(args.input, args.model_dir, args.output_dir)
    
    print(f"\n✓ Complete! Total predictions: {len(results)}")
    print(f"✓ Mutations covered: {results['mutation'].nunique()}")
