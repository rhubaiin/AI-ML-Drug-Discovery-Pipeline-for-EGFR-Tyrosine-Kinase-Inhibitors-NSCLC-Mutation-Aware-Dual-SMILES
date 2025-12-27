import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)



# Modifiying substurctures of osimertinib at specific mutation sites

#Key mutation sites in EGFR for osimertinib binding:
# 'Full_SMILES': full_smiles,
# 'ATP_POCKET': mutation_smiles, indole connects to hydrophobic back pocket, contains Cys797 region wehre Acrylamide Warhead connects to Cys797 Covalent Bond (C-S Thioether bond) 
# 'P_LOOP_HINGE': mut_hinge_p_loop, indole core connects Hydrophobic clamp Met790 gatekeeper, pyrimidine Core connects P-Loop/ Met793 hinge region respectively
# 'C_HELIX': mut_helix, dimethyl/trimethyl ethylenediamine solubility group connects partly to C-Helix salt bridge Glu762-Lys745
# 'DFG_A_LOOP': mut_dfg_a_loop,
# 'HRD_CAT': mut_hrd_cat

#Key modifications strategies to try:
#4th gen TKIs targeting C797S mutations, or bypass dependence on covalent bonding to Cys797
#Macrocyclic inhibitors and are less rotatable and more conformationally constrained, improving binding affinity and selectivity
#Larger hydrophoic groups and H bonds connecting both hydrophobic back pocket and hinge region together
#Dual warheads targeting both Cys797 and Ser797 (after C797S mutation) 

#Key AA at EGFR ATP pocket
# L718 - Bottom of ATP site, hydrophobic contact clamp
# M790 - Gatekeeper residue, hydrophobic interactions with indole ring
# M793 - Hinge region, forms H-bond with pyrimidine nitrogen
# C797 - Covalent bond with acrylamide warhead
# 19 del - activating mutation, alters local conformation of ATP pocket
# L858R - activating mutation, alters local conformation of ATP pocket
# Ex 20 ins - insertion mutations, alters local conformation of ATP pocket
# Lys745 - Salt bridge with Glu762 on C-Helix, can form H-bonds with adjacent groups
# Other residues: L788, A855, V726 



#How do positions work in BaseOsimertinibMapper and ReplacementFragmentLibrary

# BaseOsimertinibMapper numbers the base map stom_map and the base group connection points
# Eg "connection_points": {3: {"target_atom": 10, "description": "C3 indole → C4 amino pyrimidine"}}
# the 3 is from indole that connection point is 10 of the amino pyrimidine 

# the replacement group fragment             
# "methyl_indole": {"benzoamninophene": {"smiles": "Cn1cc(c2c1nccc2)","description": "N-methyl-benzoamninophene (amino heterocycle)","connection_atom": 3,"connects_to_description": "C3 position → C4 amino pyrimidine (atom 10)"}
# 1 connection point, must represent the fragment as a legit closed right by itself first, now numbered 0,1,2,3 .. then the atom number 3 of "connection_atom": 3 is the connection point, repeat for each replacement fragment
# "amino_pyrimidine": {"aminopyridimine_n_position": {"smiles": "c1cncc(n1)N",  # Complete pyrimidine ring "description": "Amino pyrimidine with swapped N position", "connection_atoms": [0, 6], "connects_to_description": "Connects to indole and phenyl"}
# 2 connection points, must define legit closed "smiles": "c1cncc(n1)N", now numbered 0, 1, 2, 3, 4, 5 then correctly count the 2 connection points of "connection_atoms": [0, 6], repeat for each replacement fragment


#Directly inserting input SMILES from alternative sources of canonical smiles, apart from fragment substitution
# Under choice no. 5
# Cn1c2c(c3ccnc(Nc4ccc(N(C)CCN(C)2)cc4NC(=O)C=C)n3)c5ccccc51 cyclic osimertinib 
# CC1C2=C(C=CC(=C2)F)C(=O)N(CC3=NN(C(=C3C4=CC(=C(N=C4)N)O1)C#N)C)C lorlatinib 
# CC(C)N1C2=CC(=NC=C2C3=C1C=C(C=C3OCC(F)(F)F)N4CCN(CC4)C)NC5=CC=NC(=N5)C6=CN(N=C6)S(=O)(=O)C7CC7 CH7233163 ()
# CO[C@@H]1CCN(C[C@@H]1F)c1nccc(Nc2cc3c(ccc(N4C[C@H](CS(C)(=O)=O)[C@H]4C)c3cn2)C(C)C)n1 BLU-945
# O=C(C1=CC(C)=NC(C2=C(OCCC[C@@H](C)C3)N(C)N=C2)=C1)/N=C(N4)/N3C5=C4C=CC(CN6CCN(C)CC6)=C5 BI-4020
#


class BaseOsimertinibMapper: 
    """Maps base osimertinib with exact atom indices and connection points."""
    #Assumptions:
    #1. base osirertinib is still the leading molecule for EGFR TKI design
    #2. Canonical smiles is sufficient to enable fragment mapping and modificatios

    def __init__(self):
        self.base_smiles = "Cn1cc(c2c1cccc2)c3ccnc(n3)Nc4cc(c(cc4OC)N(C)CCN(C)C)NC(=O)C=C"
        self.base_mol = Chem.MolFromSmiles(self.base_smiles)
        
        if self.base_mol is None:
            raise ValueError("Cannot parse base osimertinib SMILES")
        
        # Complete atom-by-atom mapping from the provided file
        self.atom_map = [
            # atom_index, original_smiles, assigned, group, connects_at
            (0, "C", "methyl indole", "methyl indole", "N1 indole"),
            (1, "n1", "N1 indole", "indole", ""),
            (2, "c", "C2 indole", "indole", ""),
            (3, "c", "C3 indole", "indole", "C4 amino pyrimidine"),
            (4, "c2", "C4", "indole", ""),
            (5, "c1", "C9", "indole", ""),
            (6, "c", "C8", "indole", ""),
            (7, "c", "C7", "indole", ""),
            (8, "c", "C6", "indole", ""),
            (9, "c2", "C5", "indole", ""),
            (10, "c3", "C4 amino pyrimidine", "amino pyrimidine", "C3 indole"),
            (11, "c", "C5 amino pyrimidine", "amino pyrimidine", ""),
            (12, "c", "C6 amino pyrimdine", "amino pyrimidine", ""),
            (13, "n", "n1 amino pyrimidine", "amino pyrimidine", ""),
            (14, "c", "C2 amino pyrimidine", "amino pyrimidine", ""),
            (15, "n3", "n3 amino pyrimidine", "amino pyrimidine", ""),
            (16, "N", "N amino pyrimidine", "amino pyrimidine", "C5 phenyl"),
            (17, "c4", "C5 phenyl", "phenyl", "N amino pyrimidine"),
            (18, "c", "C6 phenyl", "phenyl", ""),
            (19, "c", "C1 phenyl", "phenyl", "N1 acrylamide"),
            (20, "c", "C2 phenyl", "phenyl", "N1 diethylamino 1"),
            (21, "c", "C3 pheynl", "phenyl", ""),
            (22, "c4", "C4 phenyl", "phenyl", "O methoxy"),
            (23, "O", "O methoxy", "methoxy", "C4 phenyl"),
            (24, "C", "C methoxy", "methoxy", ""),
            (25, "N", "N1 diethylamino 1", "dimethyl diethylamino", "C2 phenyl"),
            (26, "C", "C1 methyl diethylamino 1", "dimethyl diethylamino", ""),
            (27, "C", "C2 diethyl", "dimethyl diethylamino", ""),
            (28, "C", "C3 diethyl", "dimethyl diethylamino", ""),
            (29, "N", "N4 diethylamino 2", "dimethyl diethylamino", ""),
            (30, "C", "C1 methyl diethylamino 2", "dimethyl diethylamino", ""),
            (31, "C", "C2 methyl diethylamino 2", "dimethyl diethylamino", ""),
            (32, "N", "N1 acrylamide", "acrylamide", "C1 phenyl"),
            (33, "C", "C1 acrylamide", "acrylamide", ""),
            (34, "O", "O1 C1 acrylamide", "acrylamide", ""),
            (35, "C", "C2 acrylamide", "acrylamide", ""),
            (36, "C", "C3 acrylamide", "acrylamide", "")
        ]
        
        # Define groups based on the atom map
        self.groups = {
            "methyl_indole": {
                "name": "Methyl Indole",
                "atoms": list(range(0, 10)),  # atoms 0-9
                "connection_points": {
                    3: {"target_atom": 10, "description": "C3 indole → C4 amino pyrimidine"}
                }
            },
            "amino_pyrimidine": {
                "name": "Amino Pyrimidine Core",
                "atoms": list(range(10, 17)),  # atoms 10-16
                "connection_points": {
                    10: {"target_atom": 3, "description": "C4 amino pyrimidine → C3 indole"},
                    16: {"target_atom": 17, "description": "N amino pyrimidine → C5 phenyl"}
                }
            },
            "phenyl": {
                "name": "Phenyl Ring",
                "atoms": list(range(17, 23)),  # atoms 17-22
                "connection_points": {
                    17: {"target_atom": 16, "description": "C5 phenyl → N amino pyrimidine"},
                    19: {"target_atom": 32, "description": "C1 phenyl → N1 acrylamide"},
                    20: {"target_atom": 25, "description": "C2 phenyl → N1 diethylamino 1"},
                    22: {"target_atom": 23, "description": "C4 phenyl → O methoxy"}
                }
            },
            "methoxy": {
                "name": "Methoxy Group",
                "atoms": [23, 24],  # atoms 23-24
                "connection_points": {
                    23: {"target_atom": 22, "description": "O methoxy → C4 phenyl"}
                }
            },
            "dimethyl_diethylamino": {
                "name": "Dimethyl Diethylamino Chain",
                "atoms": list(range(25, 32)),  # atoms 25-31
                "connection_points": {
                    25: {"target_atom": 20, "description": "N1 diethylamino 1 → C2 phenyl"}
                }
            },
            "acrylamide": {
                "name": "Acrylamide Warhead",
                "atoms": list(range(32, 37)),  # atoms 32-36
                "connection_points": {
                    32: {"target_atom": 19, "description": "N1 acrylamide → C1 phenyl"}
                }
            }
        }
    
    def display_base_map(self):
        """Display the complete base molecule atom mapping."""
        print("\n" + "="*120)
        print("BASE OSIMERTINIB ATOM MAPPING")
        print("="*120)
        print(f"Canonical SMILES: {self.base_smiles}")
        print(f"Total atoms: {self.base_mol.GetNumAtoms()}\n")
        
        print(f"{'Atom':<6} {'Original':<12} {'Assigned':<25} {'Group':<25} {'Connects at':<30}")
        print(f"{'Index':<6} {'Smiles':<12} {'':<25} {'':<25} {'':<30}")
        print("-"*120)
        
        for atom_data in self.atom_map:
            atom_idx, orig_smiles, assigned, group, connects_at = atom_data
            print(f"{atom_idx:<6} {orig_smiles:<12} {assigned:<25} {group:<25} {connects_at:<30}")
        
        print("="*120 + "\n")
        
        # Also show group summary
        print("GROUP SUMMARY:")
        print("-"*120)
        for group_key, group_info in self.groups.items():
            atom_range = f"{min(group_info['atoms'])}-{max(group_info['atoms'])}"
            print(f"\n{group_info['name']} (atoms {atom_range})")
            for atom_idx, conn_info in group_info['connection_points'].items():
                print(f"  • Atom {atom_idx}: {conn_info['description']}")
        print("="*120 + "\n")


class ReplacementFragmentLibrary:
    """Library of replacement fragments with connection point mapping."""
    # subtructure fragment indexing starts form 0
    def __init__(self):
        self.fragments = {
            #Gatekeeper Met790, the indole hydrophobic clamp interactions

            "methyl_indole": { #The hydrophobic clamp back pocket replacement options
                "benzoamninophene": {
                    "smiles": "Cn1cc(c2c1nccc2)",
                    "description": "N-methyl-benzoamninophene (amino heterocycle)",
                    "connection_atom": 3,
                    "connects_to_description": "Atom fragment 3 C3 position → C4 amino pyrimidine (atom 10)"
                },
                "pyrolfuran": {
                    "smiles": "Cn1cc(c2c1coc2)",
                    "description": "N-methyl-pyrolfuran (oxygen heterocycle)",
                    "connection_atom": 3,
                    "connects_to_description": "Atom fragment 3 C3 position → C4 amino pyrimidine (atom 10)"
                },
                "carbazole": {
                    "smiles": "Cn1c2ccccc2c3c1cccc3",
                    "description": "N-methyl-carbazole (larger bicyclic)",
                    "connection_atom": 6,
                    "connects_to_description": "Atom fragment 6 C6 → C4 amino pyrimidine (atom 10)"
                },
                "quinoline": {
                    "smiles": "Cc1ccc2ncccc2c1",
                    "description": "Methyl-quinoline",
                    "connection_atom": 3,
                    "connects_to_description": "Atom fragment 3 C3 position → C4 amino pyrimidine (atom 10)"
                },
                
                "phenothiazine": {
                    "smiles": "Cn1c2ccccc2sc3c1cccc3",
                    "description": "N-methyl-phenothiazine (tricyclic)",
                    "connection_atom": 6,
                    "connects_to_description": "Atom fragment 6 C6  → C4 amino pyrimidine (atom 10)"
                }
            },
            # Hinge region Met793, Hbonds and hydrophobic interactions

            "amino_pyrimidine": { #The hinge region hydrophobic and H-bond replacement options
                "aminopyridimine_n_position": { #shift n positioin within pyrimidine
                    "smiles": "c1cncc(n1)N",  # Complete pyrimidine ring
                    "description": "Amino pyrimidine with swapped N position",
                    "connection_atoms": [0, 6],  
                    "connects_to_description": "Atom frgament 2 → C3 indole, Atom fragement 6 → C5 phenyl"
                },
                "pyridopyrimidine": {
                    "smiles": "c1ccc2c(c1)ncc(n2)N",
                    "description": "Pyrido[3,2-d]pyrimidine",
                    "connection_atoms": [2, 10],
                    "connects_to_description": "Atom frgament 2 → C3 indole, Atom N fragement 10 → C5 phenyl"
                },
                "quinoline_core": {
                    "smiles": "c1ccc2c(c1)ccc(n2)N",
                    "description": "6-aminoquinoline core",
                    "connection_atoms": [1, 10],
                    "connects_to_description": "Atom fragment 1 → C3 indole, Atom N fragement 10 → C5 phenyl"
                },
                "benzimidazole": {
                    "smiles": "Nc1nc2ccccc2[nH]1",
                    "description": "2-aminobenzimidazole",
                    "connection_atoms": [0, 5],
                    "connects_to_description": "Atom fragment 5 → C3 indole , Atom N fragment 0 → C5 phenyl"
                },
                "purine like": {
                    "smiles": "Nc1nc2cccnc2[nH]1",
                    "description": "6-aminopurine (adenine-like)",
                    "connection_atoms": [0, 5],
                    "connects_to_description": "Atom fragment 5 → C3 indole, Atom fragment N 0 → C5 phenyl"
                },
            },

            "phenyl": {  #The phenyl ring replacement options
                "base_phenyl": { # 4 connection points, base map actually starts of c1 from amino pyrimidine. default osimertinib phenyl test for correct connections
                    "smiles": "c1ccccc1",  
                    "description": "phenyl",
                    "connection_atoms": [0, 2, 3, 5],  
                    "connects_to_description": "Atom frgament 0 c1  → C5 amino pyrimidine, Atom fragement 2 c3  → N1 acrylamide, Atom fragement 3 c4 → N1 diethylamino 1, Atom fragement 5 c6 → O methoxy"
                },
                "base_cyclpentene": { #cyclopentene base 
                    "smiles": "C1=CC=CC1",  
                    "description": "phenyl",
                    "connection_atoms": [1, 2, 3, 4],  
                    "connects_to_description": "Atom frgament 1 c2  → C5 amino pyrimidine, Atom fragement 2 c3  → N1 acrylamide, Atom fragement 3 c4 → N1 diethylamino 1, Atom fragement 4 c5 → O methoxy"
                },
                "amino_phenyl": { 
                    "smiles": "c1ncccc1",  
                    "description": "phenyl",
                    "connection_atoms": [0, 2, 3, 5],  
                    "connects_to_description": "Atom frgament 0 c1  → C5 amino pyrimidine, Atom fragement 2 c3  → N1 acrylamide, Atom fragement 3 c4 → N1 diethylamino 1, Atom fragement 5 c6 → O methoxy"
                },
                "phospo_phenyl": { #phospho group on base phenyl for potential TKI/multikinase/cytotoxic combos 
                    "smiles": "c1pcccc1",  
                    "description": "phenyl",
                    "connection_atoms": [0, 2, 3, 5],  
                    "connects_to_description": "Atom frgament 0 c1  → C5 amino pyrimidine, Atom fragement 2 c3  → N1 acrylamide, Atom fragement 3 c4 → N1 diethylamino 1, Atom fragement 5 c6 → O methoxy"
                },
                "phenyl_isomer1": {  #shift diethylamino position
                    "smiles": "c1ccccc1",  
                    "description": "phenyl",
                    "connection_atoms": [0, 2, 4, 5],  
                    "connects_to_description": "Atom frgament 0 c1  → C5 amino pyrimidine, Atom fragement 2 c3  → N1 acrylamide, Atom fragement 4 c5 → N1 diethylamino 1, Atom fragement 5 c6 → O methoxy"
                },
                "phenyl_isomer2": {  #shift acryl amide position
                    "smiles": "c1ccccc1",  
                    "description": "phenyl",
                    "connection_atoms": [0, 1, 3, 5],  
                    "connects_to_description": "Atom frgament 0 c1  → C5 amino pyrimidine, Atom fragement 1 c2  → N1 acrylamide, Atom fragement 3 c4 → N1 diethylamino 1, Atom fragement 5 c6 → O methoxy"
                },
                "base_pyrole": { #base pyrrole
                    "smiles": "c1[nH]ccc1",  
                    "description": "phenyl",
                    "connection_atoms": [0, 2, 3, 4],  
                    "connects_to_description": "Atom frgament 0 c1  → C5 amino pyrimidine, Atom fragement 2 c3  → N1 acrylamide, Atom fragement 3 c4 → N1 diethylamino 1, Atom fragement 4 c5 → O methoxy"
                },
                "base_phosphole": { ##phospho group on base cyclopentene for potential TKI/multikinase/cytotoxic combos  
                    "smiles": "P1cccc1",  
                    "description": "phenyl",
                    "connection_atoms": [1, 2, 3, 4],  
                    "connects_to_description": "Atom frgament 1 c2  → C5 amino pyrimidine, Atom fragement 2 c3  → N1 acrylamide, Atom fragement 3 c4 → N1 diethylamino 1, Atom fragement 4 c5 → O methoxy"
                },
                "base_phosphole": { ##phospho group on base phenyl for potential TKI/multikinase/cytotoxic combos  
                    "smiles": "P1cccc1",  
                    "description": "phenyl",
                    "connection_atoms": [1, 2, 3, 4],  
                    "connects_to_description": "Atom frgament 1 c2  → C5 amino pyrimidine, Atom fragement 2 c3  → N1 acrylamide, Atom fragement 3 c4 → N1 diethylamino 1, Atom fragement 4 c5 → O methoxy"
                },
             },

            "acrylamide": { #The acrylamide warhead replacement options for covalent bonding to C797S muts, new warheads are desgined to react with nucleophilic O- of S797, ideally dual warheads targeting both C797 and S797
                "amine_sulfonyl_flouride": { #possible likely for EGFR S797 only, after C797S mutation 
                    "smiles": "NS(=O)(=O)F",
                    "description": "stable sulfonate ester warhead",
                    "connection_atom": 0,
                    "connects_to_description": "N1 amine_sulfonyl_flouride → C1 phenyl (atom 19)"
                },
                "base_sulfonyl_flouride": { #dual warhead for C797 and S797 targeting, medium reactivity
                    "smiles": "S(=O)(=O)F",
                    "description": "dual warhead",
                    "connection_atom": 0,
                    "connects_to_description": "S1 sulfonyl_flouride → C1 phenyl (atom 19)"
                },
                "Fluoro_acrylamide": { #dual warhead for C797 and S797 targeting, medium reactivity
                    "smiles": "NC(=O)C(F)=C",
                    "description": "dual warhead",
                    "connection_atom": 0,
                    "connects_to_description": "N1 Fluoro_acrylamide → C1 phenyl (atom 19)"
                },
                "propargyl_amide": {
                    "smiles": "NC(=O)CC#C",
                    "description": "Propargyl amide (alkyne warhead)",
                    "connection_atom": 0,
                    "connects_to_description": "N1 acrylamide → C1 phenyl (atom 19)"
                },
                "chloroacetamide": {
                    "smiles": "NC(=O)CCl",
                    "description": "Chloroacetamide (reactive)",
                    "connection_atom": 0,
                    "connects_to_description": "N1 acrylamide → C1 phenyl (atom 19)"
                },
                "vinyl_sulfone": {# designed to reach with O- S797, Ser-O⁻ attacks sulfur 
                    "smiles": "NS(=O)(=O)C=C",
                    "description": "Vinyl sulfone warhead",
                    "connection_atom": 0,
                    "connects_to_description": "N1 acrylamide → C1 phenyl (atom 19)"
                },
                "bromoacrylamide": {
                    "smiles": "NC(=O)C(Br)=C",
                    "description": "α-bromo acrylamide",
                    "connection_atom": 0,
                    "connects_to_description": "N1 acrylamide → C1 phenyl (atom 19)"
                },
                "alpha_cyano": {
                    "smiles": "NC(=O)C(C#N)=C",
                    "description": "α-cyano acrylamide",
                    "connection_atom": 0,
                    "connects_to_description": "N1 acrylamide → C1 phenyl (atom 19)"
                },
                "maleimide": {
                    "smiles": "O=C1C=CC(=O)N1",
                    "description": "Maleimide warhead",
                    "connection_atom": 6,
                    "connects_to_description": "N1 acrylamide → C1 phenyl (atom 19)"
                },
                "vinyl_ketone": {
                    "smiles": "NC(=O)C=C",
                    "description": "Simple acrylamide (less reactive)",
                    "connection_atom": 0,
                    "connects_to_description": "N1 acrylamide → C1 phenyl (atom 19)"
                },
                "cyanoacrylamide": {
                    "smiles": "NC(=O)C(=C)C#N",
                    "description": "Cyanoacrylamide",
                    "connection_atom": 0,
                    "connects_to_description": "N1 acrylamide → C1 phenyl (atom 19)"
                }
            },
            
            "dimethyl_diethylamino": { #The solvent solubility group replacement options
                "morpholine": {
                    "smiles": "C1COCCN1",
                    "description": "Morpholine (improved solubility)",
                    "connection_atom": 5,
                    "connects_to_description": "N1 diethylamino 1 → C2 phenyl (atom 20)"
                },
                "piperazine": {
                    "smiles": "C1CNCCN1",
                    "description": "Piperazine",
                    "connection_atom": 5,
                    "connects_to_description": "N1 diethylamino 1 → C2 phenyl (atom 20)"
                },
                "n_methylpiperazine": {
                    "smiles": "CN1CCNCC1",
                    "description": "N-methyl piperazine",
                    "connection_atom": 6,
                    "connects_to_description": "N1 diethylamino 1 → C2 phenyl (atom 20)"
                },
                "pyrrolidine": {
                    "smiles": "C1CCCN1",
                    "description": "Pyrrolidine (5-membered ring)",
                    "connection_atom": 4,
                    "connects_to_description": "N1 diethylamino 1 → C2 phenyl (atom 20)"
                },
                "piperidine": {
                    "smiles": "C1CCCCN1",
                    "description": "Piperidine (6-membered ring)",
                    "connection_atom": 5,
                    "connects_to_description": "N1 diethylamino 1 → C2 phenyl (atom 20)"
                },
                "dimethylamino_propyl": {
                    "smiles": "CN(C)CCC",
                    "description": "Dimethylamino-propyl",
                    "connection_atom": 5,
                    "connects_to_description": "Atom fragment 5 C5 → C2 phenyl (atom 20)"
                },
                "diethylamino": {
                    "smiles": "CCN(CC)CC",
                    "description": "Triethylamine",
                    "connection_atom": 5,
                    "connects_to_description": "Atom fragment 5 C5→ C2 phenyl (atom 20)"
                },
                "thiomorpholine": {
                    "smiles": "C1CSCCN1",
                    "description": "Thiomorpholine (sulfur analog)",
                    "connection_atom": 5,
                    "connects_to_description": "Atom fragment 5 C5→ C2 phenyl (atom 20)"
                }
            },
            
            "methoxy": { #The methoxy group replacement options
                "ethoxy": {
                    "smiles": "CCO",
                    "description": "Ethoxy group",
                    "connection_atom": 2,
                    "connects_to_description": "O methoxy → C4 phenyl (atom 22)"
                },
                "propoxy": {
                    "smiles": "CCCO",
                    "description": "Propoxy group",
                    "connection_atom": 3,
                    "connects_to_description": "O methoxy → C4 phenyl (atom 22)"
                },
                "trifluoromethoxy": {
                    "smiles": "C(F)(F)(F)O",
                    "description": "Trifluoromethoxy (electron-withdrawing)",
                    "connection_atom": 4,
                    "connects_to_description": "O methoxy → C4 phenyl (atom 22)"
                },
                "hydroxyl": {
                    "smiles": "O",
                    "description": "Hydroxyl (H-bond donor)",
                    "connection_atom": 0,
                    "connects_to_description": "O methoxy → C4 phenyl (atom 22)"
                },
                "isopropoxy": {
                    "smiles": "CC(C)O",
                    "description": "Isopropoxy",
                    "connection_atom": 3,
                    "connects_to_description": "O methoxy → C4 phenyl (atom 22)"
                },
                "difluoromethoxy": {
                    "smiles": "C(F)(F)O",
                    "description": "Difluoromethoxy",
                    "connection_atom": 3,
                    "connects_to_description": "O methoxy → C4 phenyl (atom 22)"
                }
            }
        }
    
    #Display all replacement options for a group with connection details
    def display_group_replacements(self, group_key: str):
        
        if group_key not in self.fragments:
            print(f"Group '{group_key}' not found!")
            return None
        
        print(f"\n{'='*120}")
        print(f"REPLACEMENT OPTIONS FOR: {group_key.upper().replace('_', ' ')}")
        print(f"{'='*120}")
        
        replacements = self.fragments[group_key]
        for i, (name, info) in enumerate(replacements.items(), 1):
            print(f"\n{i}. {name}")
            print(f"   SMILES: {info['smiles']}")
            print(f"   Description: {info['description']}")
            print(f"   Connection: {info['connects_to_description']}")
            
            # Fix: Check for both 'connection_atoms' (plural) and 'connection_atom' (singular)
            if 'connection_atoms' in info:
                print(f"   Fragment connection atoms: {info['connection_atoms']}")
            elif 'connection_atom' in info:
                print(f"   Fragment connection atom: {info['connection_atom']}")
            else:
                print(f"   Fragment connection atom: Not specified")
        
        print(f"\n{'='*120}\n")
        return list(replacements.keys())


class MoleculeModifier:
    """Handles molecular modifications with connection point validation."""
    
    def __init__(self, base_mapper: BaseOsimertinibMapper, fragment_library: ReplacementFragmentLibrary):
        self.base_mapper = base_mapper
        self.fragment_library = fragment_library
    
    def replace_group(self, current_smiles: str, group_key: str, fragment_name: str) -> Tuple[Optional[str], str]:
        """
        Replace a group in the molecule with a new fragment.
        Returns: (new_smiles, status_message)
        """
        
        # Parse current molecule
        mol = Chem.MolFromSmiles(current_smiles)
        if mol is None:
            return None, "❌ Failed to parse current molecule"
        
        # Get group and fragment info
        if group_key not in self.base_mapper.groups:
            return None, f"❌ Group '{group_key}' not found"
        
        if fragment_name not in self.fragment_library.fragments[group_key]:
            return None, f"❌ Fragment '{fragment_name}' not found"
        
        group_info = self.base_mapper.groups[group_key]
        fragment_info = self.fragment_library.fragments[group_key][fragment_name]
        
        print(f"\n{'='*100}")
        print(f"REPLACING: {group_info['name']} with {fragment_name}")
        print(f"{'='*100}")
        print(f"Fragment SMILES: {fragment_info['smiles']}")
        print(f"Connection details: {fragment_info['connects_to_description']}")
        
        try:
            # Create editable molecule
            rwmol = Chem.RWMol(mol)
            
            # Step 1: Identify external connections
            atoms_to_remove = set(group_info['atoms'])
            external_connections = []
            
            for conn_atom_idx, conn_info in group_info['connection_points'].items():
                target_atom_idx = conn_info['target_atom']
                bond = rwmol.GetBondBetweenAtoms(conn_atom_idx, target_atom_idx)
                if bond:
                    external_connections.append({
                        'group_atom': conn_atom_idx,
                        'external_atom': target_atom_idx,
                        'bond_type': bond.GetBondType(),
                        'description': conn_info['description']
                    })
                    print(f"  ✓ Found connection: {conn_info['description']} (bond type: {bond.GetBondType()})")
            
            if len(external_connections) != len(group_info['connection_points']):
                return None, f"❌ Could not find all expected connections"
            
            # Step 2: Adjust external atom indices for removals
            atoms_to_remove_sorted = sorted(atoms_to_remove, reverse=True)
            for atom_idx in atoms_to_remove_sorted:
                for conn in external_connections:
                    if conn['external_atom'] > atom_idx:
                        conn['external_atom'] -= 1
            
            # Step 3: Remove old group
            for atom_idx in atoms_to_remove_sorted:
                rwmol.RemoveAtom(atom_idx)
            
            print(f"  ✓ Removed {len(atoms_to_remove)} old group atoms")
            
            # Step 4: Parse and add new fragment
            frag_mol = Chem.MolFromSmiles(fragment_info['smiles'])
            if frag_mol is None:
                return None, f"❌ Failed to parse fragment SMILES"
            
            # Map new fragment atoms
            frag_atom_map = {}
            for i in range(frag_mol.GetNumAtoms()):
                old_atom = frag_mol.GetAtomWithIdx(i)
                new_idx = rwmol.AddAtom(Chem.Atom(old_atom.GetAtomicNum()))
                frag_atom_map[i] = new_idx
                
                # Copy properties
                new_atom = rwmol.GetAtomWithIdx(new_idx)
                new_atom.SetFormalCharge(old_atom.GetFormalCharge())
                new_atom.SetNumExplicitHs(old_atom.GetNumExplicitHs())
                if old_atom.GetIsAromatic():
                    new_atom.SetIsAromatic(True)
            
            print(f"  ✓ Added {len(frag_atom_map)} new fragment atoms")
            
            # Step 5: Add internal bonds
            for bond in frag_mol.GetBonds():
                begin = frag_atom_map[bond.GetBeginAtomIdx()]
                end = frag_atom_map[bond.GetEndAtomIdx()]
                rwmol.AddBond(begin, end, bond.GetBondType())
            
            print(f"  ✓ Added internal bonds within fragment")
            
            # Step 6: Reconnect to external atoms
            conn_atoms = fragment_info.get('connection_atoms') or [fragment_info.get('connection_atom')]
            
            if len(conn_atoms) != len(external_connections):
                return None, f"❌ Mismatch: {len(conn_atoms)} fragment connections vs {len(external_connections)} required"
            
            for i, frag_conn_idx in enumerate(conn_atoms):
                new_frag_atom = frag_atom_map[frag_conn_idx]
                external_atom = external_connections[i]['external_atom']
                bond_type = external_connections[i]['bond_type']
                
                rwmol.AddBond(new_frag_atom, external_atom, bond_type)
                print(f"  ✓ Connected: fragment atom {frag_conn_idx} (mol idx {new_frag_atom}) → external atom {external_atom}")
                print(f"    ({external_connections[i]['description']})")
            
            # Step 7: Sanitize and validate
            new_mol = rwmol.GetMol()
            Chem.SanitizeMol(new_mol)
            Chem.Kekulize(new_mol, clearAromaticFlags=False)
            
            new_smiles = Chem.MolToSmiles(new_mol)
            print(f"\n  ✓✓✓ REPLACEMENT SUCCESSFUL ✓✓✓")
            print(f"  New SMILES: {new_smiles}")
            print(f"{'='*100}\n")
            
            return new_smiles, "✓ Replacement successful"
            
        except Exception as e:
            print(f"\n  ❌ REPLACEMENT FAILED: {str(e)}")
            print(f"{'='*100}\n")
            return None, f"❌ Error: {str(e)}"


class MolecularFilter:
    #Basic Molecular Filter Class for drug likeness
    
    def __init__(self):
        """Initialize all available filter catalogs from RDKit."""
        self.available_filters = {
            "Lipinski": "Rule of Five (MW≤500, LogP≤5, HBD≤5, HBA≤10)",
            "Veber": "Oral bioavailability (RotBonds≤10, TPSA≤140)",
            "Egan": "Oral drug-likeness (LogP, TPSA)",
            "Ghose": "Drug-like characteristics filter",
            "QED": "Quantitative Estimate of Drug-likeness (0-1)",
            "PAINS": "Pan Assay Interference Compounds",
            "BRENK": "Structural alerts for unwanted functionalities",
            "NIH": "NIH MLSMR excluded functionality patterns",
            "ZINC": "ZINC database filtering criteria"
        }
        
        # Initialize RDKit FilterCatalog instances
        self.catalogs = {}
        
        # PAINS filter (Pan Assay Interference)
        try:
            self.catalogs['PAINS'] = FilterCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            logger.info("✓ PAINS filter catalog loaded")
        except:
            logger.warning("✗ PAINS filter catalog unavailable")
        
        # BRENK filter (structural alerts)
        try:
            self.catalogs['BRENK'] = FilterCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            logger.info("✓ BRENK filter catalog loaded")
        except:
            logger.warning("✗ BRENK filter catalog unavailable")
        
        # NIH filter (MLSMR exclusions)
        try:
            self.catalogs['NIH'] = FilterCatalog(FilterCatalogParams.FilterCatalogs.NIH)
            logger.info("✓ NIH filter catalog loaded")
        except:
            logger.warning("✗ NIH filter catalog unavailable")
        
        # ZINC filter
        try:
            self.catalogs['ZINC'] = FilterCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
            logger.info("✓ ZINC filter catalog loaded")
        except:
            logger.warning("✗ ZINC filter catalog unavailable")
    
    def get_user_filter_selection(self) -> List[str]:
        """Get user filter selection."""
        print("\n" + "="*100)
        print("MOLECULAR FILTER SELECTION")
        print("="*100)
        for i, (name, desc) in enumerate(self.available_filters.items(), 1):
            status = "✓" if name in self.catalogs or name in ["Lipinski", "Veber", "Egan", "Ghose", "QED"] else "✗"
            print(f"{i}. [{status}] {name}: {desc}")
        
        print("\nEnter filter numbers (comma-separated, or 'all' for all available):")
        user_input = input("> ").strip().lower()
        
        if user_input == 'all':
            return list(self.available_filters.keys())
        
        try:
            choices = [int(x.strip()) for x in user_input.split(',')]
            filter_names = list(self.available_filters.keys())
            selected = [filter_names[i-1] for i in choices if 1 <= i <= len(filter_names)]
            logger.info(f"Selected filters: {', '.join(selected)}")
            return selected
        except:
            logger.warning("Invalid input, using default filters")
            return ["Lipinski", "Veber", "QED", "PAINS"]
    
    def calculate_properties(self, smiles: str) -> Dict:
        """Calculate comprehensive molecular properties."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        try:
            # Basic descriptors
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable = Lipinski.NumRotatableBonds(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            
            # Additional useful descriptors
            num_heteroatoms = Lipinski.NumHeteroatoms(mol)
            num_atoms = mol.GetNumAtoms()
            num_rings = Lipinski.RingCount(mol)
            
            # FractionCsp3 might not be available in all RDKit versions
            try:
                fraction_csp3 = Lipinski.FractionCsp3(mol)
            except AttributeError:
                # Calculate manually if not available
                num_sp3 = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3)
                fraction_csp3 = num_sp3 / num_atoms if num_atoms > 0 else 0

            return {
                'smiles': smiles,
                'molecular_weight': round(mw, 2),
                'logp': round(logp, 2),
                'hbd': hbd,
                'hba': hba,
                'tpsa': round(tpsa, 2),
                'rotatable_bonds': rotatable,
                'aromatic_rings': aromatic_rings,
                'num_heteroatoms': num_heteroatoms,
                'num_heavy_atoms': num_atoms,
                'fsp3': round(fraction_csp3, 3),
                'num_rings': num_rings,
                'qed': round(QED.qed(mol), 3)
            }
        except Exception as e:
            logger.error(f"Error calculating properties: {e}")
            return {'smiles': smiles}
    
    def apply_lipinski_filter(self, props: Dict) -> Tuple[bool, float]:
        """Lipinski Rule of Five."""
        passed = (
            props.get('molecular_weight', 501) <= 500 and
            props.get('logp', 6) <= 5 and
            props.get('hbd', 6) <= 5 and
            props.get('hba', 11) <= 10
        )
        return passed, 0.15 if passed else 0.0
    
    def apply_veber_filter(self, props: Dict) -> Tuple[bool, float]:
        """Veber rules for oral bioavailability."""
        passed = (
            props.get('rotatable_bonds', 11) <= 10 and
            props.get('tpsa', 141) <= 140
        )
        return passed, 0.15 if passed else 0.0
    
    def apply_egan_filter(self, props: Dict) -> Tuple[bool, float]:
        """Egan filter for oral drug-likeness."""
        logp = props.get('logp', 999)
        tpsa = props.get('tpsa', 999)
        
        # Egan criteria
        passed = (
            logp <= 5.88 and
            tpsa <= 131.6
        )
        return passed, 0.10 if passed else 0.0
    
    def apply_ghose_filter(self, props: Dict) -> Tuple[bool, float]:
        """Ghose filter for drug-likeness."""
        mw = props.get('molecular_weight', 0)
        logp = props.get('logp', 999)
        num_atoms = props.get('num_heavy_atoms', 0)
        
        passed = (
            160 <= mw <= 480 and
            -0.4 <= logp <= 5.6 and
            20 <= num_atoms <= 70
        )
        return passed, 0.10 if passed else 0.0
    
    def apply_qed_filter(self, props: Dict) -> Tuple[bool, float]:
        """QED (Quantitative Estimate of Drug-likeness)."""
        qed_val = props.get('qed', 0)
        passed = qed_val >= 0.5
        score = qed_val * 0.20  # QED contributes up to 20% of total score
        return passed, score
    
    def apply_catalog_filter(self, mol: Chem.Mol, catalog_name: str) -> Tuple[bool, float]:
        """Apply RDKit FilterCatalog filter."""
        if catalog_name not in self.catalogs:
            return True, 0.0
        
        try:
            catalog = self.catalogs[catalog_name]
            matches = catalog.GetMatches(mol)
            passed = len(matches) == 0
            
            # Different score weights for different catalogs
            score_weights = {
                'PAINS': 0.10,
                'BRENK': 0.08,
                'NIH': 0.07,
                'ZINC': 0.05
            }
            
            score = score_weights.get(catalog_name, 0.05) if passed else 0.0
            return passed, score
            
        except Exception as e:
            logger.error(f"Error applying {catalog_name} filter: {e}")
            return True, 0.0
    
    def score_molecule(self, smiles: str, selected_filters: List[str]) -> Dict:
        """Apply selected filters and calculate comprehensive score."""
        props = self.calculate_properties(smiles)
        
        if not props or 'molecular_weight' not in props:
            props['score'] = 0.0
            props['total_filters_passed'] = 0
            props['total_filters_applied'] = 0
            return props
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            props['score'] = 0.0
            props['total_filters_passed'] = 0
            props['total_filters_applied'] = 0
            return props
        
        total_score = 0.0
        filters_passed = 0
        filters_applied = 0
        
        # Apply each selected filter
        for filter_name in selected_filters:
            filters_applied += 1
            
            if filter_name == "Lipinski":
                passed, score = self.apply_lipinski_filter(props)
                props['lipinski_pass'] = passed
                total_score += score
                if passed:
                    filters_passed += 1
            
            elif filter_name == "Veber":
                passed, score = self.apply_veber_filter(props)
                props['veber_pass'] = passed
                total_score += score
                if passed:
                    filters_passed += 1
            
            elif filter_name == "Egan":
                passed, score = self.apply_egan_filter(props)
                props['egan_pass'] = passed
                total_score += score
                if passed:
                    filters_passed += 1
            
            elif filter_name == "Ghose":
                passed, score = self.apply_ghose_filter(props)
                props['ghose_pass'] = passed
                total_score += score
                if passed:
                    filters_passed += 1
            
            elif filter_name == "QED":
                passed, score = self.apply_qed_filter(props)
                props['qed_pass'] = passed
                total_score += score
                if passed:
                    filters_passed += 1
            
            elif filter_name in ["PAINS", "BRENK", "NIH", "ZINC"]:
                passed, score = self.apply_catalog_filter(mol, filter_name)
                props[f'{filter_name.lower()}_pass'] = passed
                total_score += score
                if passed:
                    filters_passed += 1
        
        props['score'] = round(total_score, 3)
        props['total_filters_passed'] = filters_passed
        props['total_filters_applied'] = filters_applied
        props['pass_rate'] = round((filters_passed / filters_applied * 100) if filters_applied > 0 else 0, 1)
        
        return props


class OsimertinibGenerator:
    """Main class orchestrating the molecule generation pipeline."""
    
    def __init__(self):
        self.base_mapper = BaseOsimertinibMapper()
        self.fragment_library = ReplacementFragmentLibrary()
        self.modifier = MoleculeModifier(self.base_mapper, self.fragment_library)
        self.filter = MolecularFilter()
        self.generated_molecules = []
    
    def run(self):
        """Main interactive loop."""
        print("\n" + "="*120)
        print("OSIMERTINIB ANALOG GENERATOR v3.0")
        print("Interactive Molecule Design System with Complete Atom Mapping")
        print("="*120)
        
        # Show base molecule map
        self.base_mapper.display_base_map()
        
        while True:
            print("\n" + "="*120)
            print("MAIN MENU")
            print("="*120)
            print("1. Start new molecule (from base osimertinib)")
            print("5. Add custom SMILES directly")
            print("2. View generated molecules")
            print("3. Finish and export to CSV")
            print("4. Exit without saving")
            
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == '1':
                self._design_molecule()
            elif choice == '2':
                self._view_molecules()
            elif choice == '3':
                self._export_molecules()
                break
            elif choice == '4':
                print("\nExiting without export.")
                break
            elif choice == '5': 
                self._add_custom_smiles()
            else:
                print("Invalid choice! Please enter 1-5.")
    
    def _design_molecule(self):
        """Interactive molecule design session."""
        current_smiles = self.base_mapper.base_smiles
        modifications = []
        
        print("\n" + "="*120)
        print("NEW MOLECULE DESIGN SESSION")
        print("="*120)
        print(f"Starting from base osimertinib")
        print(f"SMILES: {current_smiles}\n")
        
        while True:
            print("\n" + "-"*120)
            print("SELECT GROUP TO REPLACE:")
            print("-"*120)
            groups = list(self.base_mapper.groups.keys())
            for i, group_key in enumerate(groups, 1):
                group_info = self.base_mapper.groups[group_key]
                atom_range = f"atoms {min(group_info['atoms'])}-{max(group_info['atoms'])}"
                print(f"{i}. {group_info['name']} ({atom_range})")
            print(f"{len(groups)+1}. Done - Save this molecule")
            print(f"{len(groups)+2}. Cancel - Discard this molecule")
            
            group_choice = input("\nEnter choice: ").strip()
            
            try:
                group_idx = int(group_choice) - 1
                
                if group_idx == len(groups):  # Done
                    if modifications:
                        print(f"\n{'='*120}")
                        print(f"FINAL MOLECULE")
                        print(f"{'='*120}")
                        print(f"SMILES: {current_smiles}")
                        print(f"Modifications applied: {', '.join(modifications)}")
                        print(f"{'='*120}")
                        
                        save = input("\nSave this molecule to library? (y/n): ").strip().lower()
                        if save == 'y':
                            self.generated_molecules.append({
                                'smiles': current_smiles,
                                'modifications': ', '.join(modifications)
                            })
                            print("✓ Molecule saved to library!")
                        else:
                            print("Molecule discarded.")
                    else:
                        print("No modifications made. Returning to main menu.")
                    break
                    
                elif group_idx == len(groups) + 1:  # Cancel
                    print("Discarding molecule and returning to main menu...")
                    break
                
                elif 0 <= group_idx < len(groups):
                    group_key = groups[group_idx]
                    
                    # Show replacement options
                    fragment_names = self.fragment_library.display_group_replacements(group_key)
                    
                    if fragment_names is None:
                        continue
                    
                    frag_choice = input("\nEnter replacement number (0 to skip): ").strip()
                    
                    try:
                        frag_idx = int(frag_choice) - 1
                        
                        if frag_idx == -1:  # Skip
                            print("Skipping replacement.")
                            continue
                        
                        if 0 <= frag_idx < len(fragment_names):
                            fragment_name = fragment_names[frag_idx]
                            
                            # Attempt replacement
                            new_smiles, message = self.modifier.replace_group(
                                current_smiles, group_key, fragment_name
                            )
                            
                            if new_smiles:
                                current_smiles = new_smiles
                                modifications.append(f"{group_key}→{fragment_name}")
                                print(f"\n✓✓ Current molecule updated successfully!")
                                print(f"✓✓ Current SMILES: {current_smiles}")
                            else:
                                print(f"\n{message}")
                                print("Keeping previous structure.")
                        else:
                            print("Invalid fragment number!")
                    
                    except ValueError:
                        print("Invalid input!")
                
                else:
                    print("Invalid choice!")
                    
            except ValueError:
                print("Invalid input! Please enter a number.")
    
    def _add_custom_smiles(self):
        """Allow user to add custom SMILES directly."""
        print("\n" + "="*120)
        print("ADD CUSTOM SMILES")
        print("="*120)
        print("Enter your canonical SMILES string (or 'cancel' to return):")
        
        smiles_input = input("> ").strip()
        
        if smiles_input.lower() == 'cancel':
            print("Cancelled.")
            return
        
        # Validate the SMILES
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            print("\n❌ Invalid SMILES! Cannot parse the structure.")
            print("Please check your SMILES string and try again.")
            return
        
        # Canonicalize it
        canonical_smiles = Chem.MolToSmiles(mol)
        
        print(f"\n✓ Valid SMILES detected!")
        print(f"Original input: {smiles_input}")
        print(f"Canonical form: {canonical_smiles}")
        
        # Ask for description/modifications field
        print("\nCheck smiles press Enter to continue:")
        description = input("> ").strip()
        
        if not description:
            description = "Custom SMILES (user input)"
        
        print(f"\n{'='*120}")
        print("RUNNING MOLECULAR FILTER...")
        print(f"{'='*120}")
        
        # Get filter selection or use default
        print("\nUse default filters (Lipinski, Veber, QED, PAINS)? (y/n)")
        use_default = input("> ").strip().lower()
        
        if use_default == 'y':
            selected_filters = ["Lipinski", "Veber", "QED", "PAINS"]
            print("Using default filters...")
        else:
            selected_filters = self.filter.get_user_filter_selection()
        
        # Score the molecule
        print("\nScoring molecule...")
        scored_props = self.filter.score_molecule(canonical_smiles, selected_filters)
        
        # Display filter results
        print(f"\n{'='*120}")
        print("FILTER RESULTS")
        print(f"{'='*120}")
        print(f"SMILES: {canonical_smiles}")
        print(f"Description: {description}")
        print(f"\nMolecular Properties:")
        print(f"  Molecular Weight: {scored_props.get('molecular_weight', 'N/A')}")
        print(f"  LogP: {scored_props.get('logp', 'N/A')}")
        print(f"  HBD: {scored_props.get('hbd', 'N/A')}")
        print(f"  HBA: {scored_props.get('hba', 'N/A')}")
        print(f"  TPSA: {scored_props.get('tpsa', 'N/A')}")
        print(f"  Rotatable Bonds: {scored_props.get('rotatable_bonds', 'N/A')}")
        print(f"  QED: {scored_props.get('qed', 'N/A')}")
        print(f"\nFilter Results:")
        print(f"  Total Score: {scored_props.get('score', 0.0)}")
        print(f"  Filters Passed: {scored_props.get('total_filters_passed', 0)}/{scored_props.get('total_filters_applied', 0)}")
        print(f"  Pass Rate: {scored_props.get('pass_rate', 0)}%")
        
        # Show individual filter results
        filter_checks = [
            ('lipinski_pass', 'Lipinski'),
            ('veber_pass', 'Veber'),
            ('qed_pass', 'QED'),
            ('pains_pass', 'PAINS'),
            ('egan_pass', 'Egan'),
            ('ghose_pass', 'Ghose'),
            ('brenk_pass', 'BRENK'),
            ('nih_pass', 'NIH'),
            ('zinc_pass', 'ZINC')
        ]
        
        for key, name in filter_checks:
            if key in scored_props:
                status = "✓ PASS" if scored_props[key] else "✗ FAIL"
                print(f"  {name}: {status}")
        
        print(f"{'='*120}")
        
        # Confirm save after seeing filter results
        save = input("\nSave this molecule to library? (y/n): ").strip().lower()
        if save == 'y':
            self.generated_molecules.append({
                'smiles': canonical_smiles,
                'modifications': description
            })
            print("✓ Custom SMILES saved to library!")
        else:
            print("Molecule discarded.")
    
    # Create DataFrame with single

    def _view_molecules(self):
        """View currently generated molecules."""
        if not self.generated_molecules:
            print("\n" + "="*120)
            print("No molecules generated yet!")
            print("="*120)
            return
        
        print(f"\n{'='*120}")
        print(f"GENERATED MOLECULES LIBRARY ({len(self.generated_molecules)} total)")
        print(f"{'='*120}\n")
        
        for i, mol_info in enumerate(self.generated_molecules, 1):
            print(f"{i}. SMILES: {mol_info['smiles']}")
            print(f"   Modifications: {mol_info['modifications']}\n")
        
        print(f"{'='*120}")
    
    def _export_molecules(self):
        """Export molecules to CSV with comprehensive filtering."""
        if not self.generated_molecules:
            print("\n" + "="*120)
            print("No molecules to export!")
            print("="*120)
            return
        
        print(f"\n{'='*120}")
        print("FILTERING AND SCORING MOLECULES")
        print(f"{'='*120}")
        
        # Get user filter selection
        selected_filters = self.filter.get_user_filter_selection()
        
        print(f"\nProcessing {len(self.generated_molecules)} molecules with {len(selected_filters)} filters...\n")
        
        results = []
        for i, mol_info in enumerate(self.generated_molecules, 1):
            print(f"Scoring molecule {i}/{len(self.generated_molecules)}...", end='\r')
            scored = self.filter.score_molecule(mol_info['smiles'], selected_filters)
            scored['modifications'] = mol_info['modifications']
            results.append(scored)
        
        print()  # New line after progress
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        if 'score' in df.columns:
            df = df.sort_values('score', ascending=False, ignore_index=True)
        
        # Display summary
        print(f"\n{'='*120}")
        print("FILTERING RESULTS SUMMARY")
        print(f"{'='*120}")
        print(f"Total molecules processed: {len(df)}")
        
        if 'score' in df.columns and len(df) > 0:
            print(f"\nScore Statistics:")
            print(f"  Average score: {df['score'].mean():.3f}")
            print(f"  Score range: {df['score'].min():.3f} - {df['score'].max():.3f}")
            print(f"  Median score: {df['score'].median():.3f}")
            
            if 'pass_rate' in df.columns:
                print(f"  Average pass rate: {df['pass_rate'].mean():.1f}%")
            
            print(f"\nFilter Pass Rates:")
            
            # Display pass rates for each filter
            filter_columns = [col for col in df.columns if col.endswith('_pass')]
            for col in filter_columns:
                filter_name = col.replace('_pass', '').upper()
                pass_count = df[col].sum()
                pass_rate = (pass_count / len(df)) * 100
                print(f"  {filter_name}: {pass_count}/{len(df)} ({pass_rate:.1f}%)")
        
        # Save to file
        output_file = "osimertinib_analogs.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        
        # Display top molecules
        if len(df) > 0:
            print(f"\n{'='*120}")
            print("TOP MOLECULES (ranked by score):")
            print(f"{'='*120}\n")
            
            # Select columns to display
            priority_cols = ['smiles', 'modifications', 'score', 'pass_rate', 
                           'molecular_weight', 'logp', 'qed', 'tpsa']
            display_cols = [col for col in priority_cols if col in df.columns]
            
            n_display = min(10, len(df))
            top_molecules = df[display_cols].head(n_display)
            
            # Display with better formatting
            pd.set_option('display.max_colwidth', 50)
            pd.set_option('display.width', 120)
            print(top_molecules.to_string(index=True))
        
        print(f"\n{'='*120}")
        print("EXPORT COMPLETE")
        print(f"{'='*120}\n")


if __name__ == "__main__":
    print("\n" + "="*120)
    print("OSIMERTINIB ANALOG GENERATOR v3.0")
    print("="*120)
    print("Features:")
    print("  • Complete atom-by-atom mapping from reference file")
    print("  • Interactive group replacement with connection point validation")
    print("  • Chemical correctness checking (Sanitize + Kekulize)")
    print("  • Drug-likeness filtering (Lipinski, Veber, QED, PAINS)")
    print("  • CSV export with molecular properties")
    print("="*120 + "\n")
    
    try:
        generator = OsimertinibGenerator()
        generator.run()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\n❌ FATAL ERROR: {str(e)}")
        print("Please check the error log for details.")



#TODO
# Expand ligand Map to be more general deviating away from osimertinib core
# Add more replacement fragments for each substructure, expand library
# Screen generated ligands for druglikeness, synthetic accessibility, toxicity
# Expand to ALK, MET, ROS1, BRAF, MEK, RET inhibitors