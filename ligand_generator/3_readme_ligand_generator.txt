
Overview

1. Mechanism and Code Representation
2. Pseudocode rationale and explanation
3. Assumptions and limitations 
4. Todo
5. References

Code mechanism and representation

1. Full osimertinib atom map, substructure fragment groups, and connection points labelling. 
2. Fragment groups modifications based on user input and choice
3. Alternatively, direct full smiles based on user based input 
3. Simple molecule filter scoring for drug likeliness

Rationale and specificity of code

1. In Pseudocode 

Assumptions

1. In Pseudocode 

General Streghts
 
1. Quick filter for specific 4th generation theory inspired EGFR NSCLC mutation targets possibilities 
2. Potential to form broadspecturm multikinase/cytotoxic/ADC combinations
3. Quick tool to simulate and test future acquired resistance mutations from new generations of TKI (can mannualy induced mutations for anticipated acquired TKI resistance)

General Limitations

1. Theory inspired modifications, very specific to EGFR NSCLC TKI 
2. Does not consider EGFR receptors from autophosphorylation and dimerisation signal transductors, or other bypass mechanisms
3. Simplistic filtering drug like criteria 


TODO

1. In Pseudocode


References/Pip/Libraries/LLM

1. RDKit, Pandas, Numpy, Leskoff smiles, USCF Chimera, Pymol
2. Claude AI was used to assist with debugging, user input structure, and file reading/saving operations during code development.

Journal References

1.	Das D, Xie L, Hong J. Medicinal Chemistry. RSC Med Chem [Internet]. 2024;15:3371–94. Available from: http://dx.doi.org/10.1039/D4MD00384E
2.	Zhang D, Zhao J, Yang Y, Dai Q, Zhang N, Mi Z, et al. Fourth-generation EGFR-TKI to overcome C797S mutation : past , present , and future. J Enzyme Inhib Med Chem [Internet]. 2025;40(1). Available from: https://doi.org/10.1080/14756366.2025.2481392
3.	Grabe T, Jeyakumar K, Niggenaber J, Schulz T, Koska S, Kleinb S, et al. Addressing the Osimertinib Resistance Mutation EGFR-L858R/C797S with Reversible Aminopyrimidines. 2023; 
4.	Niggenaber J, Heyden L, Grabe T, Mu MP, Lategahn J, Rauh D. Complex Crystal Structures of EGFR with Third-Generation Kinase Inhibitors and Simultaneously Bound Allosteric Ligands. 2020; 

