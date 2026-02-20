
Overview

1. Mechanism and Code Representation for Priority Based Hierachichal LTSM GRU Custom Model 
2. Pseudocode rationale and explanation
3. Model upgrade connecting base pre-trained models
4. Assumptions 
5. Strengths & limitations 
5. Todo
6. References 

Code Mechanism & Representation 

1. End to end descriptor feature capture from both ligand and mutation protein using smiles 
2. Generates ligand and mutation  intermolecular,intramolecular, similarity, fingerprints, and custom interacting relationship features. 
2. Hierachical mechanistic priority based feature capture of custom relationship, fingerprints, similarity and molecular features. (Forward Neural Network )
3. Feature capture of subsegment smiles looped over to represent structural sequence mechanism between ligand to mutation protein in timesteps.(Recurrent Neural Network) 
4. Trained to match target activity scores (IC50,EC50,GI50) & include docking scores as second y target)

Model Upgrade connecting base pre-trained model architecture

1. Base custom model is Priority Based Hierachichal LTSM GRU custom script model architecture.
2. Model upgrade to KAN layers, Chemberta base, and GNN base. 
3. Dummy model to differentiate feautre generation and model architecture superiority. 

Rationale and specificity of code

1. In Pseudocode 

Assumptions

1. In Pseudocode

General Strenghts & Uses

1. Higher precision towards specific EGFR mutation NSCLC TKIs
2. Quick filter for specific EGFR NSCLC mutation targets possibilities  
3. Quick tool to simulate and test future acquired resistance mutations from new generations of TKI (can mannualy induced mutations for anticipated acquired TKI resistance)

General Limitations

1. Overfitting
2. Overlapping of bonding and forces on representations
3. Descriptor accuracy version on representations


TODO

1. In Pseudocode


References/Pip/Libraries/LLM/Agents
1. RDKit, Pandas, Numpy, Scikit, TF
2. Claude AI, Antigravity, VS code was used to assist with debugging, model architecture, and file reading/saving/plotting operations during code development.


Journal References

1.	Inhibitors F generation E, Chang H, Zhang Z, Tian J, Bai T, Xiao Z, et al. Machine Learning-Based Virtual Screening and Identification of the. 2024; 
2.	Lin B. A comprehensive review and comparison of existing computational methods for protein function prediction. 2024;25(4). 
3.	Hadni H, Elhallaouia M. Heliyon docking , ADMET properties and molecular dynamics simulations. Heliyon [Internet]. 2022;8(November):e11537. Available from: https://doi.org/10.1016/j.heliyon.2022.e11537
4.	Shah PM, Zhu H, Lu Z, Wang K, Tang J, Li M. DeepDTAGen : a multitask deep learning framework for drug-target af fi nity prediction and target-aware drugs generation. Nat Commun [Internet]. 2025; Available from: http://dx.doi.org/10.1038/s41467-025-59917-6
5.	Wei C, Ji C, Zong K, Zhang X, Zhong Q, Yan H, et al. Journal of Molecular Graphics and Modelling Identification of novel inhibitors targeting EGFR L858R / T790M / C797S against NSCLC by molecular docking , MD simulation , and DFT approaches. 2025;138(November 2024). 
6.	Das AP, Mathur P, Agarwal SM. Machine Learning, Molecular Docking, and Dynamics-Based Computational Identification of Potential Inhibitors against Lung Cancer. 2024; 
7.	Zhou R, Liu Z, Wu T, Pan X, Li T, Miao K, et al. Machine learning-aided discovery of T790M- mutant EGFR inhibitor CDDO-Me effectively suppresses non-small cell lung cancer growth. 2024;7. 
8.	Koh HY, Nguyen ATN, Pan S, May LT, Webb GI, Pan S, et al. PSICHIC : physicochemical graph neural network for learning protein-ligand interaction fingerprints from sequence data. 2023; 
9.	Robichaux JP, Le X, Vijayan RSK, Hicks JK, Heeke S, Elamin YY, et al. Structure-based classification predicts drug response in EGFR-mutant NSCLC. Nature [Internet]. 2021;597(7878):732–7. Available from: http://dx.doi.org/10.1038/s41586-021-03898-1


