
Overview

1. Dataset source
2. Dataset understanding, filtering, and validation
3. Dataset upgrade for substructure mutation and smiles
4. Assumptions
5. Stregths & Limitations
6. References 

Dataset Source

1. Chembl 

Dataset understanding, filtering, and validation

1. Each column is in the dataset it checked and understood
Filter and categorise to EGFR vs ADME , further EGFR to On Target and Off Target, Using On Target EGFR Only to train model

2. Filtering 
Chembl TKI EGFR NSCLC dataset ligand filter steps

Standard type column
Keep IC50 (most common standrd to show drug concentration required for 50% inhibition of autophosphorylation at TKD)
Keep EC50 (measure of anti-proliferative acitivty concentration for 50% inhibition of autophosphorylation at TKD)
Keep GI50 (dominant measure of growth inihibitor because off target signals included concentration 50% inhibition of autophosphorylation at TKD and downstream signals)

Essentially IC50 = EC50 = GI50 if IC50 is minimum threshold of acceptance 

Discard ratios, Ki, Kd, Emax and others for simplicity (ratios combine different cell host - complexity uncertain) 

Target Name Column (host cell, they harbour different mutations) 
Keep EGFR only, higher corelation for unproven ligands

Discard 
ALL humnan organ cell host (test for damage on regular human cells)
Others discarded

Target Name
EGFR Only

Target Organism column
Keep homosapiens only

Data validation 
Discard potential errors and blanks

Standard units column
Keep nM only

Assay Description & Standard Value Column only checked
(Assay Descriptions which have known mutants are favoured, uncertain or insuffiecient information are such as cell line with no mutants discarded) 

IC50 values were not filtered nor valdiate for ground truth curation, higher risk for diluted data.

Mutations
Dataset was generally validated on Assay Description,  not manually validated for each row & column. Mostly left untouched

Dataset Upgrade

1. In order to run end to end smiles format ligand and mutant protein interaction, smiles for mutant protein needs to be generated.
2. Type of mutant protein is labelled based on mutation (tkd). 
3. Mix of Verified PDB mutant Amino Acid sequence and Mannually generated mutant AA seqeunce to get smiles for full structure and substructure of mutation proteins
4. Optimised dock scores were used.
5. Manual Validation on dataset, PDB proteins, Amino Acid Seqeunces, and Smiles Strings.

Assumptions 

1. 1st generation ligands work only on sinlge mutants (del19/L858R)
2. 2nd generation ligands work only on sinlge mutants (del19/L858R)
3. 3rd generation ligands work only on single and double mutants (del19/L858R and T790M)
4. None work on triple mutants (del19/T790M/C797S or L858R/T790M/C797S)
5. Insertion exon 20 mutations are uncertain
6. None work on wild mutants

General Strenghts 
1. Large dataset 

Limitations

1. Higher uncertainty in ground truth for unproven ligands. 
2. Higher risk for dataset dilution of IC50 values duirng training process 

References/pip/libraries/llm

1. RDKit, PePSMI Novopro, USCF Chimera, Pymol 
2. Claude AI was used to assist with cleaning, debugging, hierachal categorisation, and file reading/saving operations during code development.

Journal References(1,2)
1.	Hunter FMI, Ioannidis H, Patr A, Bosc N, Corbett S, Felix E, et al. Drug and Clinical Candidate Drug Data in ChEMBL. 2025; 
2.	Zdrazil B. Fifteen years of ChEMBL and its role in cheminformatics and drug discovery. 2025; 
