
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
Keep AC50 and activity (include first verify on 2nd pass)
Else Discard
Essentially IC50 = EC50 = GI50 if IC50 is minimum threshold of acceptance 

Discard ratios, Ki, Kd, Emax and others for simplicity (ratios combine different cell host - complexity uncertain) 

Target Name Column (host cell, they harbour different mutations) 
Keep A-431 
Keep A-549
Keep Baf3
Keep calu 3 and 6 
Keep EGFR
Keep HCC827
Keep Hela 
Keep all NCI-H 1299, 1975, 3255, 460, 292
Keep PC 9
KIV mus musculus, human cell line, other general host (include first verify on 2nd pass)
Keep uncheck and blanks ((include first verify on 2nd pass)

Discard 
ALL humnan organ cell host (test for damage on regular human cells)


Target Organism column
Keep homosapiens, rats (mus musculus, rattus, cricetulus), and blanks

Data validation 
Discard potential errors

Assay type column
A includes both activity and ADME, cannot filter for now

Standard units column
Keep nM only

Smiles column
Discard salt forms

Assay Description & Standard Value Column , then manually check based on relevance 
(Assay Descriptions which have known mutants are favoured, uncertain or insuffiecient information are such as cell line with no mutants discarded) 

Dataset 1 — Validated Drug TKI Corpus 
This smaller dataset was constructed to serve as clinical ground-truth reference and evaluation benchmark, representing approved EGFR TKIs with well-characterised trial profiles. Each row was individually validated by the user, making this the more labour-intensive but higher confidence of the two corpora. Curation was guided by the following criteria applied to raw ChEMBL exports.


Data Curation and Stratified IC50 Thresholding

A multi-stage curation and stratification pipeline was implemented to construct mutation-specific training labels while minimizing assay noise and cross-target bias.

1.Single-mutant stratified IC50 thresholding.

Positive activity labels were assigned to clinically established EGFR TKIs, including erlotinib, gefitinib, afatinib, dacomitinib, osimertinib, and lazertinib. Compounds were classified as active when IC50 values were ≤100 nM. Negative labels consisted of rociletinib, ibrutinib, tigozertinib, crizotinib, ceritinib, brigatinib, and dasatinib, with IC50 values >100 nM considered inactive.

2.Double-mutant stratified IC50 thresholding.

For double-mutant EGFR variants, positive activity labels were restricted to osimertinib and lazertinib, reflecting their known clinical activity profiles. Active compounds were defined using an IC50 threshold ≤140 nM. Negative labels included dacomitinib, gefitinib, and erlotinib with IC50 values >100 nM. Afatinib activity was retained without modification to avoid ovvercorrection and recent development in the field.
 
3.Triple-mutant stratified IC50 thresholding.

No positive activity labels were assigned for triple-mutant EGFR configurations due to the absence of validated inhibitors within the curated dataset. Negative labels included gefitinib, erlotinib, afatinib, dacomitinib, osimertinib, and lazertinib, with non-working IC50 values beginning at 48.6 nM and above.

4.Extreme outlier removal and IC50 cleaning.

All records with IC50 values exceeding 10,000 nM or containing missing activity values were removed to eliminate extreme experimental outliers and incomplete measurements.

5.Bioactivity type standardization.

Only IC50, EC50, and GI50 assay types were retained. IC50 values were used as the minimum acceptance criterion, while EC50 and GI50 were considered functionally equivalent indicators of 50% inhibition at the tyrosine kinase domain (TKD).

6.Target protein and host cell annotation.

Only assays conducted in cancer cell lines harboring known EGFR mutations were included. Experiments involving off-target EGFR systems, non-EGFR assays, cytotoxicity screens, or ADME cell lines were excluded. Cell line identity was verified using the assay description fields in the curated dataset.

7.Organism scope.

Data were restricted to Homo sapiens and selected murine models (Mus musculus). Records lacking explicit organism labels were retained only if sufficient contextual annotation confirmed relevance to EGFR-targeted assays.

8.Unit standardization.

All activity values were standardized to nanomolar (nM) units, and entries reported in other units were excluded.

9.SMILES sanitization.

Chemical structures were retained in canonical isomeric SMILES format to preserve stereochemical information required for downstream machine learning representation.


Mutations
Keep Mutations that have the initial sensitising mutations and subsequent tki induced resistant mutations
Discard mutations which are primarily known as de novo resistance mutations (dilutes causality) L858R/C797S & del19/c797s discarded
Validated manually for correctness 

Dataset Upgrade

1. In order to run end to end smiles format ligand and mutant protein interaction, smiles for mutant protein needs to be generated. 
2. Type of mutant protein is labelled based on mutation (tkd).
3. Mix of Verified PDB mutant Amino Acid sequence and Mannually generated mutant AA seqeunce to get smiles for full structure and substructure of mutation proteins
4. Optimised dock scores were used.
5. Manual Validation on dataset, PDB proteins, Amino Acid Seqeunces, and Smiles Strings

Assumptions 

1. 1st generation ligands work only on sinlge mutants (del19/L858R)
2. 2nd generation ligands work only on sinlge mutants (del19/L858R)
3. 3rd generation ligands work only on single and double mutants (del19/L858R and T790M)
4. None work on triple mutants (del19/T790M/C797S or L858R/T790M/C797S)
5. Insertion exon 20 mutations are uncertain
6. None work on wild mutants

General Strengths

1. High level for ground truth on IC50 clinical relevance
2. User directed filtering to ensure less diluted training and redundancy 

General Limitations

1. User definied bias and errors in selection process 
2. Uneven dataset, unequal sample sizes for each mutation 
3. Small dataset

References/pip/libraries/llm

1. RDKit, PePSMI Novopro, USCF Chimera, Pymol 
2. Claude AI was used to assist with cleaning, debugging, hierachal categorisation, and file reading/saving operations during code development.

Journal References(1,2)
1.	Hunter FMI, Ioannidis H, Patr A, Bosc N, Corbett S, Felix E, et al. Drug and Clinical Candidate Drug Data in ChEMBL. 2025; 
2.	Zdrazil B. Fifteen years of ChEMBL and its role in cheminformatics and drug discovery. 2025; 

