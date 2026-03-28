
import os
import sys
import pickle
import numpy as np
import pandas as pd
from loguru import logger
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Multiply, LSTM, GRU, Bidirectional, LeakyReLU
from tensorflow.keras.optimizers import Adam # Needed for recompile
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from numpy.linalg import norm
from rdkit import RDLogger
# TensorFlow/Keras imports
import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
    print("TensorFlow configured to use CPU (avoiding CuDNN mismatch)")
except:
    pass

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Concatenate, 
    Multiply, LSTM, GRU, Bidirectional, LeakyReLU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

os.environ['MPLBACKEND'] = 'Agg'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(42)
torch.manual_seed(42)

# Logger configuration

logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("adv_physchem_gnn_{time}.txt", rotation="500 MB", retention="10 days", 
           compression="zip", level="DEBUG", 
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")



print("="*80)
print("MolCLR-GNN INTEGRATED HIERARCHICAL RNN LTSM MODEL")
print("Fine-tunable GNN + Priority-Gated Architecture + RNN-LSTM")
print("="*80)

# =============================================================================
# MolCLR GINet Architecture (from yuyangw/MolCLR)
# =============================================================================

#Assumptions:
#1. Adding a GNN base embedding to hierchical features at priority 1 improves performance and accuracy
#2. Adding df_train dataset ligands and mutants GNN generated embeddings without fine tuning is sufficient for initial testing
#3. Direct concatentation of GNN embeddings nodes and edges with hierarchical features preserves feature integrity
#4. The final head includes the Hierarchical and RNN-LSTM layers on top of GNN embeddings while preserving ligand mutant subtrcuture sequence 

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
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    return np.where(denominator != 0, numerator / denominator, default)

def generate_lig_inter_features(smiles): #Intermolecular Ligand, input smiles ligand, returns np array
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    
    try:
        #Hydrogen Bonding,
        features.append(Lipinski.NumHDonors(mol))
        features.append(Lipinski.NumHAcceptors(mol))
        features.append(Lipinski.NHOHCount(mol))
        features.append(Lipinski.NOCount(mol))
        features.append(rdMolDescriptors.CalcNumHBD(mol)) #includes N, O, and S (manual edit)
        features.append(rdMolDescriptors.CalcNumHBA(mol)) #includes N, O, and S (manual edit)
        
        #Electrostatic bonding
        #Partial charge = the small positive or negative charge assigned to each atom due to unequal sharing of electrons in bonds (like in polar bond
        features.append(Descriptors.MaxPartialCharge(mol)) #highest partial charge among all atoms in the molecule (most positive atom), (likely electrophilic)
        features.append(Descriptors.MinPartialCharge(mol)) #highest partial charge among all atoms in the molecule (most negative atom), (likely nucleophilic)
        features.append(Descriptors.MaxAbsPartialCharge(mol)) #largest absolute value of partial charge among all atom, strongest charge polarization within the molecule
        features.append(Descriptors.MaxPartialCharge(mol) - Descriptors.MinPartialCharge(mol)) #difference between most positive and most negative atoms), larger values mean stronger polarity within the molecule.
        features.append(Descriptors.MinAbsPartialCharge(mol))
        
        #Polar surface
        features.append(MolSurf.TPSA(mol))#The sum of the surface areas of all polar atoms (mostly oxygen and nitrogen) and their attached hydrogens.High TPSA â†’ more polar, less membrane permeable, more soluble, Low TPSA â†’ less polar, more membrane permeable (good for oral drugs)
        
        features.append(MolSurf.LabuteASA(mol))#An approximation of the total solvent-accessible surface area (SASA) of the molecule, calculated using Labuteâ€™s algorithm. #Reflects molecular size and hydrophobic surface exposure
        #Aiâ€‹=Siâ€‹â‹…Piâ€‹, S = 4Ï€r^2i , Piâ€‹=1âˆ’jâˆâ€‹(1âˆ’fij
        # S=total spherical surface area of atom , Pi=atomic solvation parameter, fij=fraction of atom i's surface area in contact with atom j

        features.append(Crippen.MolMR(mol))#The Ghose-Crippen formula is an atom-contribution method used to estimate the octanol-water partition coefficient (log P) and molar refractivity (MR) of a molecule.
        #Molar refractivity, a measure of the polarizability of the molecule
        
        #Size & Rigidity  (#May overlap with others)
        features.append(Descriptors.MolWt(mol)) #molecular weight
        features.append(Lipinski.HeavyAtomCount(mol))
        features.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        
        features.append(Crippen.MolLogP(mol))
        features.append(Descriptors.FractionCSP3(mol)) # fraction of SP3 hybridised carbons
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


def generate_mut_inter_features(smiles): #Intermolecular Mutation, input smiles mutation, returns np array
    return generate_lig_inter_features(smiles)


#Subsequent priority of descriptors to capture features from intramolecular forces 
def generate_lig_intra_features(smiles): #Intramolecular Ligand
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    
    try:
        #Covalent bond
        num_bonds = mol.GetNumBonds()
        features.append(num_bonds)
        
        #higher order bonds favours intramolecular forces within molecule, bond order indicates strength of bond
        single_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 1.0)
        double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 2.0)
        triple_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 3.0)
        aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        features.extend([single_bonds, double_bonds, triple_bonds, aromatic_bonds])
        
        avg_bond_order = np.mean([bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]) if num_bonds > 0 else 0
        features.append(avg_bond_order)
        
        #Rigidity (May Overlap), flexibility indicates less intramolecular forces within molecule
        features.append(Lipinski.NumRotatableBonds(mol))
        features.append(Lipinski.RingCount(mol))
        features.append(Lipinski.NumAromaticRings(mol))
        
        rigid_bonds = sum(1 for bond in mol.GetBonds() if bond.IsInRing())
        fraction_rigid = rigid_bonds / num_bonds if num_bonds > 0 else 0
        features.append(fraction_rigid)
        
        # Pi-Pi bonding 
        features.append(Descriptors.NumAromaticCarbocycles(mol))
        features.append(Descriptors.NumAromaticHeterocycles(mol))
        
        #Hybridization (May Overlap), branching indicates less intramolecular forces within molecule
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


def generate_mut_intra_features(smiles): # Intramolecular Mutation
    return generate_lig_intra_features(smiles)

def calculate_similarity_metrics(vec1, vec2): #input np arrays, returns a dict with math metrics
    # 1. Calculate Cosine Similarity with safety check
    norm1 = norm(vec1)
    norm2 = norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        # If either vector has zero norm, return default values
        return {
            'cosine_similarity': 0.0,
            'sine_dissimilarity': 0.0,
            'dot_product': 0.0
        }
    
    cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    sine_of_angle = np.sqrt(1 - cosine_sim**2)

    #logger.info(f"Cosine Similarity: {cosine_sim}, Sine dssimilarity: {sine_of_angle}, Dot Product: {np.dot(vec1, vec2)}")
    
    return {
        'cosine_similarity': cosine_sim,
        'sine_dissimilarity': sine_of_angle,
        'dot_product': np.dot(vec1, vec2)
    }


def calculate_fp_metrics(smiles1, smiles2): #input smiles, returns dict with rdkit datastructs similarity
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return {'dice_sim': 0.0, 'tanimato': 0.0}
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    dice_sim = DataStructs.DiceSimilarity(fp1, fp2)
    tanimato = DataStructs.TanimotoSimilarity(fp1, fp2)

    #logger.info(f"Dice Similarity: {dice_sim}, Tanimoto Similarity: {tanimato}")

    
    return {
        'dice_sim': dice_sim,
        'tanimato': tanimato,
    }


def generate_inter_interaction_features(lig_inter, mut_inter): #similarity on intermolecular interactions ligand and mutation
    features = []
    metrics = calculate_similarity_metrics(lig_inter, mut_inter)
    
    features.append(metrics['cosine_similarity'])
    features.append(metrics['sine_dissimilarity'])
    
    return np.array(features)


def generate_intra_interaction_features(lig_intra, mut_intra): #similarity on intramolecular interactions ligand and mutation
    features = []
    metrics = calculate_similarity_metrics(lig_intra, mut_intra)
    
    features.append(metrics['cosine_similarity'])
    features.append(metrics['sine_dissimilarity'])
    
    return np.array(features)


def generate_final_interaction_features(lig_smiles, mut_smiles): # fingerprints, morgan fingerprints dominate
    features = []
    
    fp_inter_metrics = calculate_fp_metrics(lig_smiles, mut_smiles)
    features.extend([fp_inter_metrics['dice_sim'], fp_inter_metrics['tanimato']])
    
    return np.array(features)


def generate_custom_features(lig_inter, mut_inter, lig_intra, mut_intra): 
    """Generate custom intermolecular and intramolecular features with safe division"""
    lig_mut_inter = []
    lig_mut_intra = []
    lig_mut_mix_inter_intra = []
    
    # H attraction ligand , H = lig_hbd . mut_hba / mut_hbd 
    # (#assumption: ligand moves to mut (mut is fixed position), lig hbd and mut hba attracts ligand), 
    # mut hbd repels favouring intra bond within mut, ignoring intra bond repelling from ligand
    H_linear_lipinski = safe_divide(lig_inter[0] * mut_inter[1], mut_inter[0], default=0.0)
    lig_mut_inter.append(H_linear_lipinski)
    
    H_linear_total = safe_divide(lig_inter[4] * mut_inter[5], mut_inter[4], default=0.0)
    lig_mut_inter.append(H_linear_total)
    
    # H attraction ligand , H = lig_hbd . mut_hba / mut_hbd with weighted mut bond path (Kappa)
    # (#assumption: ligand moves to mut (mut is fixed position), lig hbd and mut hba attracts ligand), 
    # mut hbd repels favouring intra bond within mut
    H_path = safe_divide(safe_divide(lig_inter[0] * mut_inter[1], mut_inter[0], default=0.0), mut_intra[21], default=0.0)
    lig_mut_mix_inter_intra.append(H_path)
    
    # Streght H bond in intermolecular lig to mut minus mut intra bond within mut Lig(x1,y1) Mut(x2,y2)
    # total attraction H_stregth , Lig(x1y1) Mut(x2,y2) , (lig_x1 * mut_y2 / lig_x2) + (lig_x2 * mut_y1 / mut_y2)
    # inter bond attarct x1y2 , intra bond forming assumed as repelled, x1/y1 , assumed no repelling inter H bonds
    H_strength = safe_divide(lig_inter[0] * mut_inter[1], lig_inter[1], default=0.0) + safe_divide(lig_inter[1] * mut_inter[0], mut_inter[1], default=0.0)
    lig_mut_inter.append(H_strength)
    
    H_strength_total = safe_divide(lig_inter[4] * mut_inter[5], lig_inter[4], default=0.0) + safe_divide(lig_inter[5] * mut_inter[4], mut_inter[5], default=0.0)
    lig_mut_inter.append(H_strength_total)
    
    # Lig donating stregght + Mut accepting Stregth , ligand movving to mut
    H_frac_lipinski = safe_divide(lig_inter[0], lig_inter[1], default=0.0) + safe_divide(mut_inter[1], mut_inter[0], default=0.0)
    lig_mut_inter.append(H_frac_lipinski)
    
    H_frac_total = safe_divide(lig_inter[4], lig_inter[5], default=0.0) + safe_divide(mut_inter[5], mut_inter[4], default=0.0)
    lig_mut_inter.append(H_frac_total)
    

    #using max positive and max negative charge, and length and size is simple number of bonds  (q1q2/r2)
    # Attraction opp site charge lig(q1/r1) * mut(q2/r2), q1 is max positive and q2 is max neg
    # size options include: Molwt, number of bonds, Euclidean distance . radius of gyration (rdMolDescriptors.CalcRadiusOfGyration(mol))

    # Assumption: non moving mutant, only ligand moving to mutant through attraction charge Only, (taking max abs postive and min ngeative)
    # only Attraction intermolecular forces, assuming no intrabond attraction within molecule. Assumed no repelling intermolecule same charge
    #A c_linear q1 pos to q2 neg / r1r2 
    # B c_linear q1 neg to q2 pos/r1r2
    #total & ratio

    # assuming got positive charges ligand and negative charge mut with weighted size molwt
    c_linear1_size1 = safe_divide(lig_inter[6], lig_inter[14], default=0.0) * safe_divide(mut_inter[7], mut_inter[14], default=0.0)
    lig_mut_mix_inter_intra.append(c_linear1_size1)
    
    c_linear2_size1 = safe_divide(lig_inter[7], lig_inter[14], default=0.0) * safe_divide(mut_inter[6], mut_inter[14], default=0.0)
    lig_mut_mix_inter_intra.append(c_linear2_size1)
    
    c_total = (c_linear1_size1 ** 2) + (c_linear2_size1 ** 2) #bringing out magnitude of each attarction parts
    lig_mut_mix_inter_intra.append(c_total)
    
    #difference between pos lig neg mut to neg mut pos lig
    c_diff = ((lig_inter[6]) - (mut_inter[7])) - ((mut_inter[6]) - (lig_inter[7]))
    lig_mut_inter.append(c_diff)
    
    #difference between pos lig neg mut to neg mut pos lig
    c_tpsa_diff = lig_inter[11] - mut_inter[11]
    lig_mut_inter.append(c_tpsa_diff)
    
    c_crip_logh = lig_inter[17] - mut_inter[17]
    lig_mut_inter.append(c_crip_logh)
    
    frac_tpsa_logH = safe_divide(lig_inter[11] * mut_inter[11], lig_inter[17] * mut_inter[17], default=0.0)
    lig_mut_inter.append(frac_tpsa_logH)
    
    #pi-pi stacking ratio
    pi_pi_ratio1 = safe_divide(lig_inter[21] + lig_inter[22] + mut_inter[21] + mut_inter[22], lig_intra[15] + mut_intra[15], default=0.0)
    lig_mut_mix_inter_intra.append(pi_pi_ratio1)
    
    pi_pi_ratio2 = safe_divide(lig_inter[21] + lig_inter[22] + mut_inter[21] + mut_inter[22], lig_intra[22] + mut_intra[22], default=0.0)
    lig_mut_mix_inter_intra.append(pi_pi_ratio2)
    
    #Bringing out difference between a more rigid/loose ligand 

    #double/triple bond ratio increasing
    # bond rigid total double, triple n aromatic over total num of bonds (tighter intra lig and intra mut strength as a total)
    # bond single (looser intra lig and intra mut strength)
    bond_rigid = safe_divide(lig_intra[2] + lig_intra[3] + lig_intra[4], lig_intra[0], default=0.0) + safe_divide(mut_intra[2] + mut_intra[3] + mut_intra[4], mut_intra[0], default=0.0)
    bond_single = safe_divide(lig_intra[1], lig_intra[0], default=0.0) + safe_divide(mut_intra[1], mut_intra[0], default=0.0)
    bond_diff = (bond_single - bond_rigid) ** 2
    lig_mut_intra.append(bond_diff)
    
    #spsp2/sp3 ratio
    # fraction of spsp2/sp3 between ligand and mutant
    # bigger difference indicate mutants more loose, ligands are same
    hybridisation_lig = safe_divide(lig_intra[12] + lig_intra[13], lig_intra[14] + lig_intra[12] + lig_intra[13], default=0.0)
    hybridisation_mut = safe_divide(mut_intra[12] + mut_intra[13], mut_intra[14] + mut_intra[12] + mut_intra[13], default=0.0)
    hybridisation_diff = (hybridisation_mut - hybridisation_lig) ** 2
    lig_mut_intra.append(hybridisation_diff)
    
    kappa_ratio = safe_divide(lig_intra[21], mut_intra[21], default=0.0)
    lig_mut_intra.append(kappa_ratio)
    
    return lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra

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
            lig_inter, lig_intra = _generate_lig_features(lig_smi)
            ligand_cache[lig_smi] = (lig_inter, lig_intra)
        
        # Cache mutation features
        if mut_smi in mutation_cache:
            mut_inter, mut_intra = mutation_cache[mut_smi]
        else:
            mut_inter, mut_intra = _generate_lig_features(mut_smi)
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


# =============================================================================
# Hierarchical Model with GNN Integration
# =============================================================================

def build_priority_hierarchical_model(feature_dims):
    """
    Build hierarchical model with GNN embeddings at Priority 1.
    
    GNN embeddings are added alongside fingerprint and custom features
    at the highest priority layer.
    """
    logger.info('='*80)
    logger.info('BUILDING PRIORITY HIERARCHICAL MODEL (with MolCLR GNN)')
    logger.info('='*80)
    
    # GNN embedding input (ligand + mutation concatenated)
    gnn_input = None
    if 'gnn' in feature_dims and feature_dims['gnn'] > 0:
        gnn_input = Input(shape=(feature_dims['gnn'],), name='gnn_embed')
    
    # Standard inputs
    final_fp_input = Input(shape=(feature_dims['final_fp_interaction'],), name='final_fp_interaction')
    mix_input = Input(shape=(feature_dims['lig_mut_mix_inter_intra'],), name='lig_mut_mix_inter_intra')
    inter_input = Input(shape=(feature_dims['inter_interaction'],), name='inter_interaction')
    intra_input = Input(shape=(feature_dims['intra_interaction'],), name='intra_interaction')
    mut_inter_input = Input(shape=(feature_dims['mut_inter'],), name='mut_inter')
    lig_inter_input = Input(shape=(feature_dims['lig_inter'],), name='lig_inter')
    mut_intra_input = Input(shape=(feature_dims['mut_intra'],), name='mut_intra')
    lig_intra_input = Input(shape=(feature_dims['lig_intra'],), name='lig_intra')
    
    # === PRIORITY 1: GNN + Fingerprint + Custom ===
    
    # Fingerprint branch
    fp_branch = Dense(32, kernel_initializer='he_normal', name='final_dense1')(final_fp_input)
    fp_branch = LeakyReLU(alpha=0.1, name='final_leaky1')(fp_branch)
    fp_branch = BatchNormalization(name='final_bn1')(fp_branch)
    fp_branch = Dropout(0.1, name='final_dropout1')(fp_branch)
    fp_branch = Dense(16, kernel_initializer='he_normal', name='final_dense2')(fp_branch)
    fp_branch = LeakyReLU(alpha=0.1, name='final_leaky2')(fp_branch)
    fp_emb = Dense(8, activation='tanh', name='final_embedding')(fp_branch)
    
    # Mix inter-intra branch
    mix_branch = Dense(8, kernel_initializer='he_normal', name='mix_inter_intra_dense')(mix_input)
    mix_branch = LeakyReLU(alpha=0.1, name='mix_inter_intra_leaky')(mix_branch)
    mix_emb = Dense(4, activation='tanh', name='mix_inter_intra_embedding')(mix_branch)
    
    # GNN branch (trainable projection head)
    if gnn_input is not None:
        gnn_branch = Dense(64, kernel_initializer='he_normal', name='gnn_dense1')(gnn_input)
        gnn_branch = LeakyReLU(alpha=0.1, name='gnn_leaky1')(gnn_branch)
        gnn_branch = BatchNormalization(name='gnn_bn1')(gnn_branch)
        gnn_branch = Dropout(0.1, name='gnn_dropout1')(gnn_branch)
        gnn_branch = Dense(32, kernel_initializer='he_normal', name='gnn_dense2')(gnn_branch)
        gnn_branch = LeakyReLU(alpha=0.1, name='gnn_leaky2')(gnn_branch)
        gnn_emb = Dense(8, activation='tanh', name='gnn_embedding')(gnn_branch)
        priority1_combined = Concatenate(name='priority1_combined')([fp_emb, mix_emb, gnn_emb])
    else:
        priority1_combined = Concatenate(name='priority1_combined')([fp_emb, mix_emb])
    
    # === PRIORITY 2: Inter-molecular interactions ===
    inter_branch = Dense(48, kernel_initializer='he_normal', name='inter_dense1')(inter_input)
    inter_branch = LeakyReLU(alpha=0.1, name='inter_leaky1')(inter_branch)
    inter_branch = BatchNormalization(name='inter_bn1')(inter_branch)
    inter_gate = Dense(48, activation='sigmoid', name='inter_gate')(priority1_combined)
    inter_gated = Multiply(name='inter_gating')([inter_branch, inter_gate])
    inter_gated = Dropout(0.1, name='inter_dropout')(inter_gated)
    inter_branch = Dense(24, kernel_initializer='he_normal', name='inter_dense2')(inter_gated)
    inter_branch = LeakyReLU(alpha=0.1, name='inter_leaky2')(inter_branch)
    inter_emb = Dense(12, activation='tanh', name='inter_embedding')(inter_branch)
    
    priority1_2 = Concatenate(name='priority1_2_combined')([priority1_combined, inter_emb])
    
    # === PRIORITY 3: Intra-molecular interactions ===
    intra_branch = Dense(48, kernel_initializer='he_normal', name='intra_dense1')(intra_input)
    intra_branch = LeakyReLU(alpha=0.1, name='intra_leaky1')(intra_branch)
    intra_branch = BatchNormalization(name='intra_bn1')(intra_branch)
    intra_gate = Dense(48, activation='sigmoid', name='intra_gate')(priority1_2)
    intra_gated = Multiply(name='intra_gating')([intra_branch, intra_gate])
    intra_gated = Dropout(0.1, name='intra_dropout')(intra_gated)
    intra_branch = Dense(24, kernel_initializer='he_normal', name='intra_dense2')(intra_gated)
    intra_branch = LeakyReLU(alpha=0.1, name='intra_leaky2')(intra_branch)
    intra_emb = Dense(12, activation='tanh', name='intra_embedding')(intra_branch)
    
    priority1_2_3 = Concatenate(name='priority1_2_3_combined')([priority1_2, intra_emb])
    
    # === PRIORITY 4-5: Individual inter features ===
    mut_inter_branch = Dense(32, kernel_initializer='he_normal', name='mut_inter_dense1')(mut_inter_input)
    mut_inter_branch = LeakyReLU(alpha=0.1, name='mut_inter_leaky1')(mut_inter_branch)
    mut_inter_branch = BatchNormalization(name='mut_inter_bn')(mut_inter_branch)
    mut_inter_gate = Dense(32, activation='sigmoid', name='mut_inter_gate')(priority1_2_3)
    mut_inter_gated = Multiply(name='mut_inter_gating')([mut_inter_branch, mut_inter_gate])
    mut_inter_gated = Dropout(0.1, name='mut_inter_dropout')(mut_inter_gated)
    mut_inter_branch = Dense(16, kernel_initializer='he_normal', name='mut_inter_dense2')(mut_inter_gated)
    mut_inter_branch = LeakyReLU(alpha=0.1, name='mut_inter_leaky2')(mut_inter_branch)
    mut_inter_emb = Dense(8, activation='tanh', name='mut_inter_embedding')(mut_inter_branch)
    
    lig_inter_branch = Dense(32, kernel_initializer='he_normal', name='lig_inter_dense1')(lig_inter_input)
    lig_inter_branch = LeakyReLU(alpha=0.1, name='lig_inter_leaky1')(lig_inter_branch)
    lig_inter_branch = BatchNormalization(name='lig_inter_bn')(lig_inter_branch)
    lig_inter_gate = Dense(32, activation='sigmoid', name='lig_inter_gate')(priority1_2_3)
    lig_inter_gated = Multiply(name='lig_inter_gating')([lig_inter_branch, lig_inter_gate])
    lig_inter_gated = Dropout(0.1, name='lig_inter_dropout')(lig_inter_gated)
    lig_inter_branch = Dense(16, kernel_initializer='he_normal', name='lig_inter_dense2')(lig_inter_gated)
    lig_inter_branch = LeakyReLU(alpha=0.1, name='lig_inter_leaky2')(lig_inter_branch)
    lig_inter_emb = Dense(8, activation='tanh', name='lig_inter_embedding')(lig_inter_branch)
    
    inter_combined = Concatenate(name='inter_combined')([mut_inter_emb, lig_inter_emb])
    priority1_to_5 = Concatenate(name='priority1_to_5_combined')([priority1_2_3, inter_combined])
    
    # === PRIORITY 6-7: Individual intra features ===
    mut_intra_branch = Dense(32, kernel_initializer='he_normal', name='mut_intra_dense1')(mut_intra_input)
    mut_intra_branch = LeakyReLU(alpha=0.1, name='mut_intra_leaky1')(mut_intra_branch)
    mut_intra_branch = BatchNormalization(name='mut_intra_bn')(mut_intra_branch)
    mut_intra_gate = Dense(32, activation='sigmoid', name='mut_intra_gate')(priority1_to_5)
    mut_intra_gated = Multiply(name='mut_intra_gating')([mut_intra_branch, mut_intra_gate])
    mut_intra_gated = Dropout(0.25, name='mut_intra_dropout')(mut_intra_gated)
    mut_intra_branch = Dense(16, kernel_initializer='he_normal', name='mut_intra_dense2')(mut_intra_gated)
    mut_intra_branch = LeakyReLU(alpha=0.1, name='mut_intra_leaky2')(mut_intra_branch)
    mut_intra_emb = Dense(8, activation='tanh', name='mut_intra_embedding')(mut_intra_branch)
    
    lig_intra_branch = Dense(32, kernel_initializer='he_normal', name='lig_intra_dense1')(lig_intra_input)
    lig_intra_branch = LeakyReLU(alpha=0.1, name='lig_intra_leaky1')(lig_intra_branch)
    lig_intra_branch = BatchNormalization(name='lig_intra_bn')(lig_intra_branch)
    lig_intra_gate = Dense(32, activation='sigmoid', name='lig_intra_gate')(priority1_to_5)
    lig_intra_gated = Multiply(name='lig_intra_gating')([lig_intra_branch, lig_intra_gate])
    lig_intra_gated = Dropout(0.25, name='lig_intra_dropout')(lig_intra_gated)
    lig_intra_branch = Dense(16, kernel_initializer='he_normal', name='lig_intra_dense2')(lig_intra_gated)
    lig_intra_branch = LeakyReLU(alpha=0.1, name='lig_intra_leaky2')(lig_intra_branch)
    lig_intra_emb = Dense(8, activation='tanh', name='lig_intra_embedding')(lig_intra_branch)
    
    intra_combined = Concatenate(name='intra_combined')([mut_intra_emb, lig_intra_emb])
    
    # === Final Integration ===
    all_combined = Concatenate(name='all_combined')([priority1_2_3, inter_combined, intra_combined])
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
    
    # Embedding layer for RNN — tanh bounds outputs to [-1,1] for LSTM/GRU stability
    embedding_output = Dense(16, activation='tanh', kernel_initializer='glorot_uniform', name='embedding_output')(x)
    
    # Output heads
    activity_head = Dense(8, kernel_initializer='he_normal', name='activity_head')(embedding_output)
    activity_head = LeakyReLU(alpha=0.1, name='activity_head_activation')(activity_head)
    activity_head = Dropout(0.2, name='activity_head_dropout')(activity_head)
    activity_output = Dense(1, activation='linear', name='activity_output')(activity_head)
    
    docking_head = Dense(8, kernel_initializer='he_normal', name='docking_head')(embedding_output)
    docking_head = LeakyReLU(alpha=0.1, name='docking_head_activation')(docking_head)
    docking_head = Dropout(0.2, name='docking_head_dropout')(docking_head)
    docking_output = Dense(1, activation='linear', name='docking_output')(docking_head)
    
    # Build model
    inputs = []
    if gnn_input is not None:
        inputs.append(gnn_input)
    inputs.extend([final_fp_input, mix_input, inter_input, intra_input,
                   mut_inter_input, lig_inter_input, mut_intra_input, lig_intra_input])
    
    model = Model(inputs=inputs, outputs=[activity_output, docking_output],
                  name='priority_hierarchical_model_gnn')
    model.compile(
        optimizer=Adam(learning_rate=0.003),
        loss={'activity_output': 'mse', 'docking_output': 'mse'},
        loss_weights={'activity_output': 1.0, 'docking_output': 0.6},
        metrics={'activity_output': ['mae'], 'docking_output': ['mae']}
    )
    logger.info(f'Model compiled. Total params: {model.count_params():,}')
    return model


def build_rnn_sequential_model(embedding_dim, n_timesteps=6):
    """Build RNN-LSTM model for sequential mutation embeddings."""
    sequence_input = Input(shape=(n_timesteps, embedding_dim), name='mutation_sequence')
    
    # BiLSTM path
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, 
                                   recurrent_dropout=0.2), name='bilstm_1')(sequence_input)
    lstm_out = BatchNormalization(name='bn_lstm1')(lstm_out)
    lstm_out = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2,
                                   recurrent_dropout=0.2), name='bilstm_2')(lstm_out)
    lstm_out = BatchNormalization(name='bn_lstm2')(lstm_out)
    
    # BiGRU path
    gru_out = Bidirectional(GRU(128, return_sequences=True, dropout=0.2,
                                 recurrent_dropout=0.2), name='bigru_1')(sequence_input)
    gru_out = BatchNormalization(name='bn_gru1')(gru_out)
    gru_out = Bidirectional(GRU(64, return_sequences=False, dropout=0.2,
                                 recurrent_dropout=0.2), name='bigru_2')(gru_out)
    gru_out = BatchNormalization(name='bn_gru2')(gru_out)
    
    # Combine paths
    combined = Concatenate(name='lstm_gru_combined')([lstm_out, gru_out])
    x = Dense(128, activation='relu', name='rnn_dense1')(combined)
    x = BatchNormalization(name='rnn_bn1')(x)
    x = Dropout(0.3, name='rnn_dropout1')(x)
    x = Dense(64, activation='relu', name='rnn_dense2')(x)
    x = BatchNormalization(name='rnn_bn2')(x)
    x = Dropout(0.2, name='rnn_dropout2')(x)
    x = Dense(32, activation='relu', name='rnn_dense3')(x)
    x = Dropout(0.1, name='rnn_dropout3')(x)
    
    # Output heads
    activity_final = Dense(16, activation='relu', name='activity_final_head')(x)
    activity_final = Dropout(0.15, name='activity_final_dropout')(activity_final)
    activity_output = Dense(1, activation='linear', name='final_activity_output')(activity_final)
    
    docking_final = Dense(16, activation='relu', name='docking_final_head')(x)
    docking_final = Dropout(0.15, name='docking_final_dropout')(docking_final)
    docking_output = Dense(1, activation='linear', name='final_docking_output')(docking_final)
    
    model = Model(inputs=sequence_input, outputs=[activity_output, docking_output],
                  name='rnn_sequential_model')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'final_activity_output': 'mse', 'final_docking_output': 'mse'},
        loss_weights={'final_activity_output': 1.0, 'final_docking_output': 0.7},
        metrics={'final_activity_output': ['mae'], 'final_docking_output': ['mae']}
    )
    return model


# =============================================================================
# Main Workflow
# =============================================================================

def keras_main(output_dir='.', train_data_path=None, control_data_path=None, drug_data_path=None):
    """Full training workflow with predictions on control and drug datasets."""
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    print("\n" + "="*80)
    print("FULL TRAINING MODE")
    print("="*80)

    # Load datasets
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    if train_data_path is None:
        train_data_path = os.path.join(data_dir, 'df_3_shuffled.csv')
    if control_data_path is None:
        control_data_path = os.path.join(data_dir, 'egfr_tki_valid_cleaned.csv')
    if drug_data_path is None:
        drug_data_path = os.path.join(data_dir, 'drugs.csv')

    print("\nLoading datasets...")
    df_train = pd.read_csv(train_data_path, encoding='latin-1')
    df_control = pd.read_csv(control_data_path, encoding='latin-1')
    df_drugs = pd.read_csv(drug_data_path, encoding='latin-1')
    
    df_train.columns = df_train.columns.str.strip()
    
    # Filter valid training samples
    valid_mask = ~(
        df_train['smiles'].isna() |
        df_train['smiles_full_egfr'].isna() |
        df_train['smiles 718_862_atp_pocket'].isna() |
        df_train['smiles_p_loop'].isna() |
        df_train['smiles_c_helix'].isna() |
        df_train['smiles_l858r_a_loop_dfg_motif'].isna() |
        df_train['smiles_catalytic_hrd_motif'].isna() |
        df_train['standard value'].isna() |
        df_train['dock'].isna()
    )
    df_train_valid = df_train[valid_mask].reset_index(drop=True)

    # Save unique mutation profiles — required by prediction script
    _profile_cols = [
        'smiles_full_egfr', 'smiles 718_862_atp_pocket', 'smiles_p_loop',
        'smiles_c_helix', 'smiles_l858r_a_loop_dfg_motif',
        'smiles_catalytic_hrd_motif', 'tkd'
    ]
    mutation_profiles = (
        df_train_valid[_profile_cols]
        .drop_duplicates(subset=['tkd'])
        .reset_index(drop=True)
    )
    mutation_profiles.to_csv('mutation_profiles.csv', index=False)
    print(f"OK Saved {len(mutation_profiles)} mutation profiles → mutation_profiles.csv")

    print(f"Training samples: {len(df_train_valid)}/{len(df_train)}")
    print(f"Control samples: {len(df_control)}")
    print(f"Drug samples: {len(df_drugs)}")
    
    if len(df_train_valid) == 0:
        print("ERROR: No valid training samples!")
        return
    
    # Initialize GNN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    gnn_model = GINet(num_layer=5, emb_dim=300, feat_dim=512)
    gnn_model.to(device)
    gnn_model.eval()
    logger.debug("GNN GINet model initialized and set to eval mode")
    print("OK GINet model initialized")
    

    pretrained_path = "checkpoints/molclr_pretrained.pth"
    if os.path.exists(pretrained_path):
        logger.info(f"Loading MolCLR pre-trained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        gnn_model.load_pretrained(state_dict)
        logger.success("MolCLR pre-trained weights loaded successfully!")
    else:
        logger.warning(f"Pre-trained weights not found at {pretrained_path}")
        logger.warning("Continuing with randomly initialized GNN weights. "
                       "Set pretrained_path correctly for production use.")

    # Save the GNN weights used (pretrained or random) so the prediction script
    # loads the identical weights and its gnn_embedding_scalers remain valid
    torch.save(gnn_model.state_dict(), 'gnn_pretrained.pth')
    logger.info("GNN weights saved → gnn_pretrained.pth")
        
    # Extract ligand GNN embeddings
    print("\nExtracting GNN embeddings for training ligands...")
    lig_smiles_list = df_train_valid['smiles'].astype(str).tolist()
    lig_gnn_embs = get_gnn_embeddings(lig_smiles_list, gnn_model, device)
    print(f"  Ligand embeddings: {lig_gnn_embs.shape}")
    
    # Define mutation sites
    mutation_sites = [
        ('FULL_SMILES', 'smiles_full_egfr'),
        ('ATP_POCKET', 'smiles 718_862_atp_pocket'),
        ('P_LOOP_HINGE', 'smiles_p_loop'),
        ('C_HELIX', 'smiles_c_helix'),
        ('DFG_A_LOOP', 'smiles_l858r_a_loop_dfg_motif'),
        ('HRD_CAT', 'smiles_catalytic_hrd_motif')
    ]
    
    # Collect common valid indices across all sites
    all_feature_dicts = []
    all_valid_indices = []
    
    for site_name, site_col in mutation_sites:
        print(f"\n--- Generating features for {site_name} ---")
        feature_dict = generate_hierarchical_features(df_train_valid['smiles'], df_train_valid[site_col])
        all_feature_dicts.append(feature_dict)
        all_valid_indices.append(set(feature_dict['valid_indices']))
    
    common_valid = sorted(list(set.intersection(*all_valid_indices)))
    print(f"\nCommon valid samples: {len(common_valid)}")
    
    if len(common_valid) == 0:
        print("ERROR: No common valid samples!")
        return
    
    # Filter features to common indices
    for i, fd in enumerate(all_feature_dicts):
        mask = np.isin(fd['valid_indices'], common_valid)
        for key in ['lig_inter', 'mut_inter', 'inter_interaction', 'lig_intra',
                    'mut_intra', 'intra_interaction', 'lig_mut_mix_inter_intra', 'final_fp_interaction']:
            all_feature_dicts[i][key] = fd[key][mask]
    
    # Prepare targets
    y1 = np.log1p(df_train_valid['standard value'].iloc[common_valid].values)
    y2 = df_train_valid['dock'].iloc[common_valid].values
    y_scaler1 = StandardScaler()
    y_scaler2 = StandardScaler()
    y1_scaled = y_scaler1.fit_transform(y1.reshape(-1, 1)).flatten()
    y2_scaled = y_scaler2.fit_transform(y2.reshape(-1, 1)).flatten()
    
    # Training storage
    all_scalers = []
    all_gnn_scalers = []
    all_embeddings = []
    all_histories = []
    
    # Train hierarchical models for each site
    for site_idx, (site_name, site_col) in enumerate(mutation_sites):
        print(f"\n{'='*80}\nTraining Site {site_idx+1}/6: {site_name}\n{'='*80}")
        
        feature_dict = all_feature_dicts[site_idx]
        
        # Scale features
        scalers = {}
        scaled = {}
        for key in ['lig_inter', 'mut_inter', 'inter_interaction', 'lig_intra',
                    'mut_intra', 'intra_interaction', 'lig_mut_mix_inter_intra', 'final_fp_interaction']:
            scalers[key] = StandardScaler()
            scaled[key] = scalers[key].fit_transform(feature_dict[key])
        all_scalers.append(scalers)
        
        # Get mutation GNN embeddings and concatenate
        mut_smiles = df_train_valid[site_col].iloc[common_valid].astype(str).tolist()
        mut_gnn_embs = get_gnn_embeddings(mut_smiles, gnn_model, device)
        print(f"  Mutation embeddings: {mut_gnn_embs.shape}")
        lig_gnn_common = lig_gnn_embs[common_valid]
        gnn_concat = np.concatenate([lig_gnn_common, mut_gnn_embs], axis=1)
        
        gnn_scaler = StandardScaler()
        gnn_scaled = gnn_scaler.fit_transform(gnn_concat)
        all_gnn_scalers.append(gnn_scaler)
        
        # Build model
        feature_dims = {
            'gnn': gnn_scaled.shape[1],
            'final_fp_interaction': scaled['final_fp_interaction'].shape[1],
            'lig_mut_mix_inter_intra': scaled['lig_mut_mix_inter_intra'].shape[1],
            'inter_interaction': scaled['inter_interaction'].shape[1],
            'intra_interaction': scaled['intra_interaction'].shape[1],
            'mut_inter': scaled['mut_inter'].shape[1],
            'lig_inter': scaled['lig_inter'].shape[1],
            'mut_intra': scaled['mut_intra'].shape[1],
            'lig_intra': scaled['lig_intra'].shape[1],
        }
        
        model = build_priority_hierarchical_model(feature_dims)
        
        # Training callbacks
        checkpoint = ModelCheckpoint(f'gnn_hierarchical_{site_name}.h5', 
                                     monitor='val_loss', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=30, 
                                   restore_best_weights=True, verbose=1)
        
        # Train
        history = model.fit(
            x=[gnn_scaled, scaled['final_fp_interaction'], scaled['lig_mut_mix_inter_intra'],
               scaled['inter_interaction'], scaled['intra_interaction'],
               scaled['mut_inter'], scaled['lig_inter'], scaled['mut_intra'], scaled['lig_intra']],
            y={'activity_output': y1_scaled, 'docking_output': y2_scaled},
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        all_histories.append(history.history)
        
        # Extract embeddings
        emb_model = Model(inputs=model.inputs, outputs=model.get_layer('embedding_output').output)
        embs = emb_model.predict([
            gnn_scaled, scaled['final_fp_interaction'], scaled['lig_mut_mix_inter_intra'],
            scaled['inter_interaction'], scaled['intra_interaction'],
            scaled['mut_inter'], scaled['lig_inter'], scaled['mut_intra'], scaled['lig_intra']
        ], verbose=0)
        all_embeddings.append(embs)
        print(f"OK {site_name} complete - Embeddings: {embs.shape}")
        logger.debug(f"Site {site_name} training complete with embeddings shape: {embs.shape}")
    
    # Train RNN model
    print(f"\n{'='*80}\nTraining RNN-LSTM Model\n{'='*80}")
    sequential_embs = np.stack(all_embeddings, axis=1)
    print(f"Sequential embeddings: {sequential_embs.shape}")
    
    rnn_model = build_rnn_sequential_model(embedding_dim=16, n_timesteps=6)
    
    rnn_checkpoint = ModelCheckpoint('gnn_rnn_model.h5', monitor='val_loss', 
                                     save_best_only=True, verbose=1)
    rnn_early_stop = EarlyStopping(monitor='val_loss', patience=40, 
                                   restore_best_weights=True, verbose=1)
    
    rnn_history = rnn_model.fit(
        x=sequential_embs,
        y={'final_activity_output': y1_scaled, 'final_docking_output': y2_scaled},
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[rnn_checkpoint, rnn_early_stop],
        verbose=1
    )
    
    print("OK RNN training complete")
    
    # Save scalers
    with open('gnn_feature_scalers.pkl', 'wb') as f:
        pickle.dump(all_scalers, f)
    with open('gnn_y_scalers.pkl', 'wb') as f:
        pickle.dump({'y_scaler1': y_scaler1, 'y_scaler2': y_scaler2}, f)
    with open('gnn_embedding_scalers.pkl', 'wb') as f:
        pickle.dump(all_gnn_scalers, f)
    
    # Save training plot
    print("\nSaving training plot...")
    try:
        plt.figure(figsize=(14, 6))

        # Left: hierarchical per-site losses
        plt.subplot(1, 2, 1)
        for i, h in enumerate(all_histories):
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
        plt.title('Hierarchical per-site training (GNN)')
        plt.legend(fontsize='small', loc='upper right')

        # Right: RNN training loss
        plt.subplot(1, 2, 2)
        r_epochs = range(1, len(rnn_history.history.get('loss', [])) + 1)
        plt.plot(r_epochs, rnn_history.history.get('loss', []), label='RNN train loss')
        plt.plot(r_epochs, rnn_history.history.get('val_loss', []), linestyle='--', label='RNN val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('RNN training (GNN integrated)')
        plt.legend()

        plt.tight_layout()
        out_png = 'gnn_training_history.png'
        plt.savefig(out_png, dpi=200)
        print(f'Saved training history plot to {out_png}')
    except Exception as e:
        print('Could not generate training plot:', e)
    
    # Predictions on control and drugs
    predict_on_datasets(df_control, df_drugs, gnn_model, device, all_scalers, 
                        all_gnn_scalers, y_scaler1, y_scaler2, rnn_model, 
                        mutation_sites, df_train_valid)


def predict_on_datasets(df_control, df_drugs, gnn_model, device, all_scalers,
                        all_gnn_scalers, y_scaler1, y_scaler2, rnn_model,
                        mutation_sites, df_train_valid):
    """Generate predictions on control and drug datasets."""
    
    # Get unique mutation profiles
    profile_cols = ['smiles_full_egfr', 'smiles 718_862_atp_pocket', 'smiles_p_loop',
                    'smiles_c_helix', 'smiles_l858r_a_loop_dfg_motif', 
                    'smiles_catalytic_hrd_motif', 'tkd']
    unique_profiles = df_train_valid[profile_cols].drop_duplicates(subset=['tkd']).reset_index(drop=True)
    print(f"\nUnique mutation profiles: {len(unique_profiles)}")
    
    for dataset_name, df, smiles_col, name_col in [
        ('Control', df_control, 'smiles_control' if 'smiles_control' in df_control.columns else 'smiles', 
         'id' if 'id' in df_control.columns else None),
        ('Drugs', df_drugs, 'smiles', 'name' if 'name' in df_drugs.columns else None)
    ]:
        print(f"\n{'='*80}\nPredicting {dataset_name}\n{'='*80}")
        results = []
        
        compound_smiles = df[smiles_col].astype(str).tolist()
        compound_names = df[name_col].tolist() if name_col else list(range(len(df)))
        compound_gnn = get_gnn_embeddings(compound_smiles, gnn_model, device)
        
        for _, profile in unique_profiles.iterrows():
            mutation_name = profile['tkd']
            profile_embeddings = []
            valid_idx = None
            
            for site_idx, (site_name, site_col) in enumerate(mutation_sites):
                mut_smi = profile[site_col]
                mut_inter, mut_intra = _generate_lig_features(mut_smi)
                
                if mut_inter is None or mut_intra is None:
                    break
                
                features = {k: [] for k in ['lig_inter', 'mut_inter', 'inter_interaction',
                                            'lig_intra', 'mut_intra', 'intra_interaction',
                                            'lig_mut_mix_inter_intra', 'final_fp_interaction']}
                site_valid = []
                
                for idx, comp_smi in enumerate(compound_smiles):
                    lig_inter, lig_intra = _generate_lig_features(comp_smi)
                    if lig_inter is None or lig_intra is None:
                        continue
                    
                    lig_mut_inter, lig_mut_intra, lig_mut_mix = generate_custom_features(
                        lig_inter, mut_inter, lig_intra, mut_intra)
                    inter_int = generate_inter_interaction_features(lig_inter, mut_inter)
                    intra_int = generate_intra_interaction_features(lig_intra, mut_intra)
                    
                    if lig_mut_inter:
                        inter_int = np.concatenate([np.array(lig_mut_inter), inter_int])
                    if lig_mut_intra:
                        intra_int = np.concatenate([np.array(lig_mut_intra), intra_int])
                    
                    fp_int = generate_final_interaction_features(comp_smi, mut_smi)
                    
                    features['lig_inter'].append(lig_inter)
                    features['mut_inter'].append(mut_inter)
                    features['inter_interaction'].append(inter_int)
                    features['lig_intra'].append(lig_intra)
                    features['mut_intra'].append(mut_intra)
                    features['intra_interaction'].append(intra_int)
                    features['lig_mut_mix_inter_intra'].append(np.array(lig_mut_mix))
                    features['final_fp_interaction'].append(fp_int)
                    site_valid.append(idx)
                
                if len(site_valid) == 0:
                    break
                
                if valid_idx is None:
                    valid_idx = site_valid
                
                scalers = all_scalers[site_idx]
                scaled = {k: scalers[k].transform(np.array(features[k])) for k in features}
                
                mut_gnn = get_gnn_embeddings([mut_smi] * len(site_valid), gnn_model, device)
                comp_gnn = compound_gnn[site_valid]
                gnn_concat = np.concatenate([comp_gnn, mut_gnn], axis=1)
                gnn_scaled = all_gnn_scalers[site_idx].transform(gnn_concat)
                
                hier_model = load_model(f'gnn_hierarchical_{site_name}.h5', compile=False)
                emb_model = Model(inputs=hier_model.inputs, 
                                  outputs=hier_model.get_layer('embedding_output').output)
                
                embs = emb_model.predict([
                    gnn_scaled, scaled['final_fp_interaction'], scaled['lig_mut_mix_inter_intra'],
                    scaled['inter_interaction'], scaled['intra_interaction'],
                    scaled['mut_inter'], scaled['lig_inter'], scaled['mut_intra'], scaled['lig_intra']
                ], verbose=0)
                profile_embeddings.append(embs)
            
            if len(profile_embeddings) != 6 or valid_idx is None:
                continue
            
            seq_embs = np.stack(profile_embeddings, axis=1)
            preds = rnn_model.predict(seq_embs, verbose=0)
            
            activity = np.expm1(y_scaler1.inverse_transform(preds[0]))
            docking = y_scaler2.inverse_transform(preds[1])
            
            for i, idx in enumerate(valid_idx):
                results.append({
                    'compound': compound_names[idx],
                    'smiles': compound_smiles[idx],
                    'mutation': mutation_name,
                    'predicted_activity': float(activity[i][0]),
                    'predicted_docking': float(docking[i][0])
                })
        
        output_file = f'gnn_{dataset_name.lower()}_predictions.csv'
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"OK Saved {output_file} ({len(results)} rows)")
    
    print("\n" + "="*80)
    print("TRAINING AND PREDICTIONS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MolCLR-GNN Hierarchical Model (Corrected)')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save all outputs')
    parser.add_argument('--train_data', type=str, default=None, help='Training CSV path')
    parser.add_argument('--control_data', type=str, default=None, help='Control compounds CSV path')
    parser.add_argument('--drug_data', type=str, default=None, help='Drug compounds CSV path')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    parser.add_argument('--n_samples', type=int, default=5, help='Test samples')
    args = parser.parse_args()

    if args.test:
        success = test_mode(n_samples=args.n_samples)
        sys.exit(0 if success else 1)
    else:
        keras_main(output_dir=args.output_dir, train_data_path=args.train_data,
                   control_data_path=args.control_data, drug_data_path=args.drug_data)

#TODO
# 1. Conduct fine tuning using generated embedddings from of ligand and mutants in df_train dataset 
# 2. Connect the GNN embeddings precisely to intermolecular and intramolecuar features off the hierachicaal features
# 3. Experiment with different GNN pre-trained models and architectures 

