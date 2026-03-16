
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

Dataset 2 — Non-Drug Investigational TKI Corpus 

This larger corpus was assembled to provide broad chemical diversity across non-approved and investigational TKI scaffolds. The filtering strategy was deliberately less restrictive than Dataset 1, trading individual-row validation for expanded chemical coverage. Curation proceeded as follows.
Modeling Assumptions for EGFR Mutation-Specific Activity

To preserve assay fidelity while maintaining adequate chemical diversity, several assumptions were applied during the data curation and filtering process.

1.IC50 thresholding and cleaning.

Only extreme outlier values and incomplete records were removed during preprocessing. Specifically, IC50 values exceeding 10,000 nM and entries containing missing activity measurements were discarded. Valid EGFR tyrosine kinase inhibitors originating from Dataset 1 were retained in this dataset without additional IC50 filtering. As a result, the dataset intentionally preserves a wider range of reported IC50 values, acknowledging that experimentally validated inhibitors may appear across both low and high activity ranges. This design re-introduces controlled uncertainty in the classification of positive and negative labels.

2.Target protein and host cell annotation.

Only assays performed in cancer cell lines known to harbor EGFR mutations were retained to ensure biological relevance to on-target EGFR inhibition. Assays involving off-target systems, non-EGFR models, cytotoxicity screens, or ADME-related cell lines were excluded. Cell line identity and target relevance were verified using the assay description field within the curated dataset.

3.Bioactivity measurement retention.

Only IC50, EC50, and GI50 assay types were included. IC50 values served as the primary acceptance threshold, while EC50 and GI50 measurements were treated as functionally equivalent indicators of 50% inhibition at the EGFR tyrosine kinase domain (TKD).

4.Organism scope.

Data were restricted to assays conducted in Homo sapiens. Entries lacking explicit organism annotations were retained only when sufficient contextual information confirmed their relevance to human EGFR experimental systems.

5.Unit standardization.

All activity values were standardized to nanomolar (nM) units. Entries reported in other measurement units were excluded to maintain consistency across the dataset.

6.SMILES sanitization.

Chemical structures were retained in isomeric SMILES format to preserve stereochemical information required for accurate molecular representation in downstream machine learning models.


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
