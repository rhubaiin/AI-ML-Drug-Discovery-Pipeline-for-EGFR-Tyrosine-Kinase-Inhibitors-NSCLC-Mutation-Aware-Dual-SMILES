import pandas as pd
import re
import numpy as np

# =====================================================
# EGFR Dataset Filtering and Classification Script
# =====================================================

# Load the dataset
df = pd.read_csv(r'd:\Publications\project_insilico\dataset\curate_validation_set_todo_jan2026\validationset_tofilter.csv')

print(f"Original dataset shape: {df.shape}")

# =====================================================
# PART 1: DATA CLEANING
# =====================================================

def clean_text(text):
    """Clean and lowercase text"""
    if pd.isna(text):
        return ''
    return str(text).lower().strip()

# Clean all text columns
print("\n[1/6] Cleaning data...")
for col in df.columns:
    if df[col].dtype == 'object' and col.lower() != 'smiles':
        df[col] = df[col].apply(clean_text)

# Focus on the Assay Description column
df['Assay Description'] = df['Assay Description'].fillna('')

print(f"Cleaned dataset shape: {df.shape}")

# =====================================================
# PART 2: DEFINE FILTER PATTERNS
# =====================================================

# ===== EGFR ON-TARGET PATTERNS =====
egfr_patterns = [
    r'epidermal.*growth.*factor.*receptor',
    r'egfr', r'erbb1', r'erbB1',
    r'apoptosis', r'proliferation', r'cytotoxicity', r'antiproliferative', r'inihibition',
    r'anticancer', r'anti-cancer', r'antitumor',
    r'anti.*tumor', r'anti.*tumour',
    r'double.*mutant', r'triple.*mutant',
    r'egfr.*mutant', r'egfr.*variant',
    r'\b(?:tyrosine\s+)?kinase\s+domain',
    r'\btkd\b',
    r'\bkinase\s+domain',
    r'\bcatalytic\s+domain',
    r'\batp.*bind',
    r'activation.*loop',
    r'c.?lobe',
    r'n.?lobe',
    r'phosphorylation',
    r'autophosphorylation'
]

egfr_mutations = [
    r'l858r', r't790m', r'exon.*19', r'exon.*21',
    r'\b19\b', r'\b21\b', r'\b20\b',
    r'\b858\b', r'\b790\b', r'\b797\b', r'\b719\b', r'\b768\b', r'\b861\b', r'\b718\b',
    r'g719[sxac]', r'l861q', r's768i', r'c797[sx]',
    r'del19', r'e746.a750', r'l747.e749', r'l747.p753',
    r'l747.s752', r'l747.t751'
]

egfr_cell_lines = [
    r'a.?431', r'hcc4006', r'u.?87', r'u87.?mg', r'mda', r'mcf',
    r'erbB1', r'erbb1', r'hcc', r'nci', r'pc', r'calu', r'baf3', r'ba/f3', r'cal', r'bt', r'colo', r'cor', r'mcf',
    r'\b431\b', r'\b1975\b', r'\b3255\b', r'\b1650\b', r'\b827\b', r'\b4006\b',
    r'\b2279\b', r'\b9er\b', r'\b1299\b', r'\b292\b', r'\b460\b',
    r'bid007', r'kyse270', r'kyse450',
    r'nci.?h1666', r'nci.?h1975', r'hcc827', r'nci.?h3255',
    r'pc.?9', r'nci.?h1650', r'cl97', r'calu.?3',
    r'nci.?h820', r'hcc827gr5', r'pc9.?brc1', r'nci.?h2279',
    r'dfci.?127',
]

# ===== CYTOTOXICITY / ADME-TOX =====
cyto_cells = [
    r'hepatocyte', r'cho', r'hk.?2', r'l02', r'huvec', r'bj', r'ea\.hy\.?926',
    r'beas.?2b', r'hepg2', r'ht.?29', r'kg.?1', r'lymphoma',
    r'mda.?mb.?468', r'mda.?mb.?231', r'vero', r'vero.?c1008',
    r'liver', r'primary.*hepatocyte', r'hep.*rg', r'hepatocytes', r'histone.*deacetylase',
    r'guinea.*pig', r'canine', r'dog', r'monkey', r'cynomolgus',
    r'a549', r'mcf7', r'sk.?br.?3', r't47d', r'bt.?474', r'hct.?116', r'lovo',
    r'dld.?1', r'sw480', r'sw.?620', r'panc.?1', r'ovcar.?8', r'sk.?ov.?3', r'kb',
    r'a101d', r'a498', r'nb69', r'rmg.?i', r'l5178y', r'u2os', r'saos.?2', r'sjsa.?1'
]

cyto_markers = [
    r'adme', r'admet',
    r'herg', r'potassium.*channel', r'cardiac.*ion.*channel', r'ion.*channel',
    r'cytochrome.*p450', r'cyp', r'cyp1a2', r'cyp2c9', r'cyp2d6', r'cyp2c8', r'cl97', r'lsd1',
    r'cyp3a4', r'cyp2b6', r'cyp2c19', r'cyp3a5', r'drug.*metabolism', r'mgh',
    r'monoamine.*oxidase', r'mao', r'cyclooxygenase', r'cox.?1', r'cox.?2',
    r'no.*relevant.*target', r'non.?protein.*target', r'dipeptidyl.*peptidase',
    r'molecular.*identity.*unknown', r'unspecific', r'off.?target', r'histone',
    r'beta.?1.*adrenergic.*receptor', r'beta.?2.*adrenergic.*receptor', r'prostaglandin.*receptor',
    r'type.?1.*angiotensin.*ii.*receptor', r'fibroblast.*growth.*factor.*receptor', r'gamma.*aminobutyric.*acid.*receptor',
    r'adenosine.*a1.*receptor', r'adenosine.*a2a.*receptor', r'adenosine.*a3.*receptor',
    r'5.?ht', r'serotonin.*receptor', r'dopamine.*receptor', r'histamine.*receptor',
    r'opioid.*receptor', r'sodium.*channel', r'calcium.*channel', r'gaba.*receptor',
    r'glutamate.*receptor', r'nmda.*receptor', r'ampa.*receptor', r'adregenic.*receptor',
]

# ===== OFF-TARGET ACTIVITY =====
other_activity_targets = [
    r'pi3.?kinase', r'pi3k', r'pi3', r'akt', r'mtor', r'phosphoinositide.*3.?kinase',
    r'tyrosine.*protein.*kinase.*abl', r'bcr.?abl', r'bcr/abl',
    r'stem.*cell.*growth.*factor.*receptor', r'c.?kit', r'\bkit\b',
    r'fibroblast.*growth.*factor.*receptor', r'fgfr1', r'fgfr2', r'fgfr3', r'fgfr4',
    r'hepatocyte.*growth.*factor.*receptor', r'\bmet\b', r'c.?met',
    r'tyrosine.*protein.*kinase.*receptor.*ret', r'\bret\b',
    r'tyrosine.*protein.*kinase.*receptor.*flt3', r'flt3',
    r'tyrosine.*protein.*kinase.*jak', r'jak1', r'jak2', r'jak3',
    r'eml4.?alk', r'\balk\b', r'anaplastic.*lymphoma.*kinase',
    r'vascular', r'vegf', r'pdgf', r'von',
    r'serine/threonine.*protein.*kinase.*b.?raf', r'braf', r'b.?raf', r'v600e',
    r'gak', r'chk1', r'chk2', r'plk', r'polo.*like.*kinase', r'aurora.*kinase',
    r'insulin.*receptor', r'insulin.*like.*growth.*factor.*i.*receptor', r'igf.?1r',
    r'vegf.*receptor', r'vegfr', r'pdgfr', r'src', r'syk', r'btk', r'tek', r'tie2',
    r'ros1', r'trk', r'ntrk', r'dna.*pk', r'mapk', r'erk', r'mek', r'raf', r'ras',
    r'kras', r'kras.*g12c', r'kras.*g12d', r'kras.*g12v', r'hras', r'nras',
    r'c.?src', r'lck', r'fyn', r'yes', r'fgr', r'lyn', r'hck', r'blk', r'yrk',
    r'fak', r'paxillin', r'p38', r'jnk', r'sapk', r'ikk', r'tbk1', r'ikkε',
    r'androgen.*receptor', r'estrogen.*receptor', r'progesterone.*receptor',
    r'beta.*adrenergic.*receptor', r'dopamine.*receptor',
    r'glucocorticoid.*receptor', r'mineralocorticoid.*receptor',
    r'peroxisome.*proliferator.*activated.*receptor', r'ppar',
    r'quinolone.*resistance.*protein.*nora', r'venezuelan.*equine.*encephalitis.*virus',
    r'cdc42', r'bc', r'cgmp', r'hacat', r'aspc1',
]

# =====================================================
# PART 3: PATTERN MATCHING FUNCTION
# =====================================================

def pattern_match(text, patterns):
    """Check if any pattern matches the text"""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

# =====================================================
# PART 4: CATEGORIZE ASSAYS
# =====================================================

print("\n[2/6] Categorizing assays...")

def categorize_assay(row):
    """Categorize assays into: 0=unclassified, 1=EGFR on-target, 2=cytotoxicity, 3=off-target"""
    desc = row['Assay Description']
    
    # Check cytotoxicity first (higher priority)
    if pattern_match(desc, cyto_cells) or pattern_match(desc, cyto_markers):
        return 2
    
    # Check off-target
    if pattern_match(desc, other_activity_targets):
        return 3
    
    # Check EGFR on-target
    if (pattern_match(desc, egfr_patterns) or 
        pattern_match(desc, egfr_mutations) or 
        pattern_match(desc, egfr_cell_lines)):
        return 1
    
    # Unclassified
    return 0

# Apply categorization
df['Category'] = df.apply(categorize_assay, axis=1)

# =====================================================
# PART 5: CREATE FILTERED DATAFRAMES
# =====================================================

print("\n[3/6] Creating filtered DataFrames...")

df_egfr_pure = df[df['Category'] == 1].copy()
df_cyto = df[df['Category'] == 2].copy()
df_off_target = df[df['Category'] == 3].copy()
df_unclassified = df[df['Category'] == 0].copy()

print(f"EGFR on-target: {len(df_egfr_pure)} rows")
print(f"Cytotoxicity: {len(df_cyto)} rows")
print(f"Off-target: {len(df_off_target)} rows")
print(f"Unclassified: {len(df_unclassified)} rows")

# =====================================================
# PART 6: TKD CLASSIFICATION FOR EGFR_PURE
# =====================================================

print("\n[4/6] Classifying EGFR TKD mutations...")

def classify_tkd_mutation(row):
    """
    Classify EGFR mutations into specific categories with comprehensive pattern matching:
    1. del - only deletion 19 (single mutation)
    2. del/t790m double - deletion 19 + t790m (double mutant)
    3. del/t790m/c797s triple - deletion 19 + t790m + c797s (triple mutant)
    4. subs l858r - only l858r substitution (single mutation)
    5. l858r/t790m double - l858r + t790m (double mutant)
    6. l858r/t790m/c797s triple - l858r + t790m + c797s (triple mutant)
    7. ins 20 - only insertion 20 (single mutation)
    8. wild adeno lung - wild type
    0. other/unclassified
    """
    
    desc = row['Assay Description']
    variant_mutation = str(row.get('Assay Variant Mutation', '')).lower()
    
    # Combine all relevant text
    combined_text = f"{desc} {variant_mutation}"
    
    # ===== TRIPLE MUTANT PATTERNS (check first - most specific) =====
    
    # L858R/T790M/C797S triple mutant patterns
    l858r_t790m_c797s_patterns = [
        r'l858r[/\s\-_]*t790m[/\s\-_]*c797s',
        r't790m[/\s\-_]*l858r[/\s\-_]*c797s',
        r'c797s[/\s\-_]*l858r[/\s\-_]*t790m',
        r'c797s[/\s\-_]*t790m[/\s\-_]*l858r',
        r'egfr\s+l858r[/\s\-_]*t790m[/\s\-_]*c797s',
        r'triple.*mutant.*l858r.*t790m.*c797s',
        r'triple.*mutant.*t790m.*l858r.*c797s',
    ]
    
    # Del19/T790M/C797S triple mutant patterns
    del19_t790m_c797s_patterns = [
        r'del.*19[/\s\-_]*t790m[/\s\-_]*c797s',
        r't790m[/\s\-_]*del.*19[/\s\-_]*c797s',
        r'deletion.*19[/\s\-_]*t790m[/\s\-_]*c797s',
        r'e746.?a750.*t790m.*c797s',
        r'exon.*19.*deletion.*t790m.*c797s',
        r'egfr\s+del.*19[/\s\-_]*t790m[/\s\-_]*c797s',
        r'triple.*mutant.*del.*19.*t790m.*c797s',
        r'triple.*mutant.*exon.*19.*deletion.*t790m.*c797s',
    ]
    
    # Check triple mutants
    for pattern in l858r_t790m_c797s_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return 'l858r/t790m/c797s triple'
    
    for pattern in del19_t790m_c797s_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return 'del/t790m/c797s triple'
    
    # ===== DOUBLE MUTANT PATTERNS (check second) =====
    
    # L858R/T790M double mutant patterns
    l858r_t790m_patterns = [
        r'l858r[/\s\-_]*t790m(?![/\s\-_]*c797s)',  # L858R/T790M but NOT followed by C797S
        r't790m[/\s\-_]*l858r(?![/\s\-_]*c797s)',  # T790M/L858R but NOT followed by C797S
        r'egfr[:\s\-_]+l858r[/\s\-_]*t790m',
        r'egfr[:\s\-_]+t790m[/\s\-_]*l858r',
        r'egfr.?t790m.?l858',  # EGFR-T790M/L858
        r'double.*mutant.*l858r.*t790m',
        r'double.*mutant.*t790m.*l858r',
    ]
    
    # Del19/T790M double mutant patterns
    del19_t790m_patterns = [
        r'del.*19[/\s\-_]*t790m(?![/\s\-_]*c797s)',  # Del19/T790M but NOT followed by C797S
        r't790m[/\s\-_]*del.*19(?![/\s\-_]*c797s)',  # T790M/Del19 but NOT followed by C797S
        r'deletion.*19.*t790m(?![/\s\-_]*c797s)',
        r't790m.*exon.*19.*deletion(?![/\s\-_]*c797s)',
        r'e746.?a750.*t790m(?![/\s\-_]*c797s)',
        r'exon.*19.*deletion.*exon.*20.*t790m',
        r'egfr\s+del.*19[/\s\-_]*t790m',
        r'egfr\s+e746.?a750.*t790m',
        r'double.*mutant.*del.*19.*t790m',
        r'double.*mutant.*exon.*19.*deletion.*t790m',
    ]
    
    # Check double mutants
    for pattern in l858r_t790m_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return 'l858r/t790m double'
    
    for pattern in del19_t790m_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return 'del/t790m double'
    
    # ===== SINGLE MUTATION PATTERNS =====
    
    # Detection flags for single mutations
    has_del19 = bool(re.search(r'del(?:etion)?[:\s\-_]*19|exon[:\s\-_]*19[:\s\-_]*del(?:etion)?|e746.?a750|l747', combined_text, re.IGNORECASE))
    has_l858r = bool(re.search(r'l858r|\b858r', combined_text, re.IGNORECASE))
    has_t790m = bool(re.search(r't790m|\b790m', combined_text, re.IGNORECASE))
    has_c797s = bool(re.search(r'c797s|\b797s', combined_text, re.IGNORECASE))
    has_ins20 = bool(re.search(r'ins(?:ertion)?[:\s\-_]*20|exon[:\s\-_]*20[:\s\-_]*ins(?:ertion)?', combined_text, re.IGNORECASE))
    has_wild = bool(re.search(r'\bwild[:\s\-_]*type|\bwt\b|wild.*adeno', combined_text, re.IGNORECASE))
    
    # Count mutations to ensure single mutation
    mutation_count = sum([has_del19, has_l858r, has_t790m, has_c797s, has_ins20])
    
    # Single mutations
    if has_del19 and mutation_count == 1:
        return 'del'
    
    if has_l858r and mutation_count == 1:
        return 'subs l858r'
    
    if has_ins20 and mutation_count == 1:
        return 'ins 20'
    
    # Wild type (no mutations)
    if has_wild and mutation_count == 0:
        return 'wild adeno lung'
    
    # Unclassified
    return 'other'

# Apply TKD classification to EGFR pure dataset
df_egfr_pure['tkd'] = df_egfr_pure.apply(classify_tkd_mutation, axis=1)

# Count TKD categories
tkd_counts = df_egfr_pure['tkd'].value_counts()
print("\nTKD Mutation Distribution:")
for tkd_type, count in tkd_counts.items():
    print(f"  {tkd_type}: {count}")

# =====================================================
# PART 7: SAVE RESULTS
# =====================================================

print("\n[5/6] Saving results...")

output_dir = 'filtered_output'
import os
os.makedirs(output_dir, exist_ok=True)

# Save main category dataframes
df.to_csv(f'{output_dir}/00_all_categorized.csv', index=False)
df_egfr_pure.to_csv(f'{output_dir}/01_egfr_ontarget.csv', index=False)
df_cyto.to_csv(f'{output_dir}/02_cytotoxicity.csv', index=False)
df_off_target.to_csv(f'{output_dir}/03_off_target.csv', index=False)
df_unclassified.to_csv(f'{output_dir}/04_unclassified.csv', index=False)

# =====================================================
# SUMMARY REPORT
# =====================================================

print("\n" + "="*60)
print("FILTERING SUMMARY REPORT")
print("="*60)
print(f"\nTotal records processed: {len(df)}")
print(f"\n--- Category Distribution ---")
print(f"EGFR On-Target:  {len(df_egfr_pure):5d} ({len(df_egfr_pure)/len(df)*100:5.2f}%)")
print(f"Cytotoxicity:    {len(df_cyto):5d} ({len(df_cyto)/len(df)*100:5.2f}%)")
print(f"Off-Target:      {len(df_off_target):5d} ({len(df_off_target)/len(df)*100:5.2f}%)")
print(f"Unclassified:    {len(df_unclassified):5d} ({len(df_unclassified)/len(df)*100:5.2f}%)")

print(f"\n--- TKD Mutation Distribution (EGFR On-Target) ---")
for tkd_type, count in tkd_counts.items():
    pct = count/len(df_egfr_pure)*100 if len(df_egfr_pure) > 0 else 0
    print(f"{tkd_type:30s} {count:5d} ({pct:5.2f}%)")

print(f"\n--- Output Files ---")
print(f"All files saved to: {output_dir}/")
print(f"\nMain category files:")
print(f"  - 00_all_categorized.csv (with Category column)")
print(f"  - 01_egfr_ontarget.csv (with Category and tkd columns)")
print(f"  - 02_cytotoxicity.csv (with Category column)")
print(f"  - 03_off_target.csv (with Category column)")
print(f"  - 04_unclassified.csv (with Category column)")
print("\n" + "="*60)
print("FILTERING COMPLETE!")
print("="*60)
