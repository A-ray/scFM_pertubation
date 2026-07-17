# Python Script to run ISP using scGPT
# Usage: python run_isp.py [liver|immune]

# Chunk 9A
import scanpy as sc
import anndata as ad
from scipy import sparse
from scipy.stats import median_abs_deviation
import os
import sys
import numpy as np
import pandas as pd
import torch
import scgpt as scg
from pathlib import Path
import traceback
import warnings
warnings.filterwarnings('ignore')

# ── Parse command-line argument ────────────────────────────────────────────────
VALID_DATA_TYPES = ["liver", "immune"]
VALID_LABELS = ["Healthy", "MASLD", "MASH"]
VALID_PERTURB_MODES = ["down", "up"]

if len(sys.argv) != 5:
    print("Usage: python run_isp.py [liver|immune] [Initial_Label] [Target_Label] [down|up]")
    print("Example: python run_isp.py liver Healthy MASH down")
    sys.exit(1)

DATA_TYPE = sys.argv[1].lower()
INITIAL_LABEL = sys.argv[2]
TARGET_LABEL = sys.argv[3]
PERTURB_MODE = sys.argv[4].lower()

if DATA_TYPE not in VALID_DATA_TYPES:
    print(f"Error: DATA_TYPE must be one of {VALID_DATA_TYPES}")
    sys.exit(1)

if INITIAL_LABEL not in VALID_LABELS:
    print(f"Error: INITIAL_LABEL must be one of {VALID_LABELS}")
    sys.exit(1)

if TARGET_LABEL not in VALID_LABELS:
    print(f"Error: TARGET_LABEL must be one of {VALID_LABELS}")
    sys.exit(1)

if PERTURB_MODE not in VALID_PERTURB_MODES:
    print(f"Error: PERTURB_MODE must be one of {VALID_PERTURB_MODES}")
    sys.exit(1)

print(f"Running ISP for: {DATA_TYPE}")
print(f"Initial Label: {INITIAL_LABEL}")
print(f"Target Label: {TARGET_LABEL}")
print(f"Perturb Mode: {PERTURB_MODE}")

# ── scGPT ISP config ───────────────────────────────────────────────────────────
SCGPT_MODEL_DIR     = Path("/mnt/hd1/home/ankitray/scFM/scGPT_CP")   # same model dir you used for embeddings
GENE_COL            = "Gene Symbol"             # adata.var column with gene symbols
INITIAL_LABEL       = INITIAL_LABEL
TARGET_LABEL        = TARGET_LABEL
CONDITION_COL       = "broad_condition"
DONOR_COL           = "patient_id"

# Set CELL_TYPE_COL and other parameters based on DATA_TYPE
if DATA_TYPE == 'liver':
    CELL_TYPE_COL = "typist_liver_majority_voting"
    CANDIDATE_GENES_PATH = "/mnt/hd1/home/ankitray/scFM/data/candidate_gene.csv"
    DATA_DIR = Path("/mnt/hd1/home/ankitray/scFM/data/liver_cell_type_adata_embedded")
    PASSING_CELLTYPES = [
        'Macrophages', 'Cholangiocytes', 'Resident NK', 'T cells',
        'Endothelial cells', 'Hepatocytes', 'Fibroblasts'
    ]
    OUTPUT_BASE = '/mnt/hd1/home/ankitray/scFM/results/scGPT/separation_test_output_celltype_liver/{}_{}/{}/'.format(INITIAL_LABEL, TARGET_LABEL, PERTURB_MODE)
else:  # immune
    CELL_TYPE_COL = "typist_immune_majority_voting"
    CANDIDATE_GENES_PATH = "/mnt/hd1/home/ankitray/scFM/data/candidate_gene.csv"
    DATA_DIR = Path("/mnt/hd1/home/ankitray/scFM/data/immune_cell_type_adata_embedded")
    PASSING_CELLTYPES = [
        'Memory B cells', 'NK cells', 'Naive B cells', 'Tem_Trm cytotoxic T cells',
        'MAIT cells', 'CD16+ NK cells', 'CRTAM+ gamma-delta T cells', 'pDC',
        'Non-classical monocytes', 'Classical monocytes', 'CD16- NK cells',
        'Tem_Temra cytotoxic T cells', 'Tcm_Naive helper T cells', 'Tem_Effector helper T cells'
    ]
    OUTPUT_BASE = '/mnt/hd1/home/ankitray/scFM/results/scGPT/separation_test_output_celltype_immune/{}_{}/{}'.format(INITIAL_LABEL, TARGET_LABEL, PERTURB_MODE)

EMBEDDING_KEY   = 'X_scGPT'          # adata.obsm key
MIN_CELLS_PER_STATE = 30
ALT_STATES      = []                  # no alt states
OUTPUT_DIR      = Path(OUTPUT_BASE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ISP_CELLS_TOTAL     = 500    # control cells per cell type fed into ISP
MIN_CELLS_PER_DONOR_ISP = 5      # donors with fewer cells are excluded
MIN_CELLS_PER_GENE      = 20     # min cells a gene must appear in to be scored
EFFECT_SIZE_THRESHOLD   = 2.5e-4 # minimum |median cosine shift| to be considered
# For down-regulation: set raw count → 0  (embed_data bins this → bin 0)
# For up-regulation: no fixed scalar — we set raw count to 10× per-cell max
#   so the gene dominates rank-based binning and lands in bin 51.
# (PERTURB_BIN_VALUE is kept only for the log message below)
if PERTURB_MODE == 'down':
    PERTURB_BIN_VALUE = 0   # raw count 0  → bin 0  (silenced)
elif PERTURB_MODE == 'up':
    PERTURB_BIN_VALUE = None  # handled per-cell in the ISP loop (→ bin 51)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

ISP_OUTPUT_DIR   = OUTPUT_DIR / "scgpt_isp_output"
STATS_OUTPUT_DIR = OUTPUT_DIR / "scgpt_isp_stats"
os.makedirs(ISP_OUTPUT_DIR,   exist_ok=True)
os.makedirs(STATS_OUTPUT_DIR, exist_ok=True)

# ── Load candidate genes ───────────────────────────────────────────────────────
MASH_GENES = pd.read_csv(CANDIDATE_GENES_PATH, header=None)[0].tolist()

# MASH_GENES = MASH_GENES = [

#     # ── GWAS / genetic risk (10) ──────────────────────────────────────────────
#     # Validated GWAS loci — expression is context-dependent, not constitutive.
#     # Independent of expression datasets — strongest unbiased evidence.
#     "PNPLA3",    # rs738409 I148M — largest MASH risk locus; lipid droplet
#     "TM6SF2",    # E167K — VLDL secretion defect; steatosis + fibrosis
#     "MBOAT7",    # rs641738 — phosphatidylinositol remodelling
#     "HSD17B13",  # rs72613567 — loss-of-function PROTECTIVE against MASH
#     "GCKR",      # rs1260326 — glucokinase regulator; de novo lipogenesis
#     "SAMM50",    # rs3761472 — mitochondrial membrane; fibrosis GWAS
#     "NCAN",      # rs2228603 — hepatic steatosis GWAS
#     "ABCB4",     # rs2109505 — biliary phosphatidylcholine; fibrosis risk
#     "CIDEB",     # rs1805081 — lipid droplet fusion; protective variant
#     "PTPN11",    # rs11066301 — protein tyrosine phosphatase; MASH GWAS

#     # ── ECM / fibrosis / HSC activation (12) ─────────────────────────────────
#     # Near-absent in healthy liver; specifically upregulated in activated HSCs.
#     "COL1A1",    # fibrillar collagen — canonical fibrosis hallmark
#     "COL1A2",
#     "COL3A1",    # reticular collagen
#     "ACTA2",     # α-SMA — activated HSC marker; absent in quiescent HSCs
#     "LOXL1",     # collagen crosslinking; M5 fibrosis hub (Piras 2024)
#     "LOXL2",
#     "TIMP1",     # MMP inhibitor — blocks fibrosis resolution
#     "TIMP2",
#     "MMP2",      # matrix metalloproteinase — ECM remodelling
#     "MMP9",
#     "PDGFRB",    # PDGF receptor β — HSC proliferation; upregulated on activation
#     "SMOC2",     # M5 fibrosis coexpression module driver (Piras & DiStefano 2024)

#     # ── TGF-β / SMAD fibrogenic signalling (5) ───────────────────────────────
#     "TGFB1",     # master pro-fibrotic cytokine; regulated not constitutive
#     "TGFB2",
#     "SMAD2",     # canonical fibrogenic effector
#     "SMAD3",
#     "TGFBR1",    # ALK5 — TGF-β type I receptor

#     # ── Innate immune / Kupffer cells / NLRP3 (12) ───────────────────────────
#     # All inducible — not constitutively expressed at high levels.
#     "TNF",       # TNFα — inducible; central MASH pro-inflammatory cytokine
#     "IL6",       # inducible; elevated in MASH
#     "IL1B",      # NLRP3 product; specifically upregulated in MASH
#     "IL18",      # NLRP3 product; elevated in MASH
#     "CXCL10",    # IP-10 — IFN-induced; markedly elevated in MASH
#     "CCL2",      # MCP-1 — monocyte recruitment; induced in MASH
#     "CCR2",      # MCP-1 receptor on infiltrating monocytes
#     "CD68",      # macrophage marker; regulated by activation state
#     "TLR4",      # LPS receptor; key innate immune MASH driver
#     "MYD88",     # central TLR adaptor; regulated
#     "NLRP3",     # inflammasome sensor — key MASH driver; therapeutic target
#     "CASP1",     # inflammasome effector caspase; IL-1β/IL-18 maturation

#     # ── Lipid-associated macrophage (LAM) markers (5) ────────────────────────
#     # MASH-specific macrophage subpopulation (Guilliams 2022, 2025).
#     # Specifically elevated in MASH — not expressed in healthy liver macrophages.
#     "TREM2",     # LAM hub; strongly elevated in MASH macrophages
#     "GPNMB",     # glycoprotein NMB; LAM marker; fibrosis correlate
#     "SPP1",      # osteopontin; LAM/scar macrophage; MASH progression marker
#     "FABP4",     # lipid-associated macrophage marker; disease-regulated
#     "LGALS3",    # galectin-3; macrophage activation; fibrosis biomarker

#     # ── Adaptive immunity — T cells / checkpoints (4) ────────────────────────
#     "CD8A",      # cytotoxic T cell marker
#     "FOXP3",     # Treg master TF; regulated by immune context
#     "PDCD1",     # PD-1 — T cell exhaustion; elevated in chronic MASH
#     "IFNG",      # IFN-γ — inducible Th1 cytokine; Kupffer cell priming

#     # ── Lipid metabolism — specifically dysregulated in MASH (9) ─────────────
#     # Regulated by nutritional/hormonal state — not constitutively maximal.
#     "FASN",      # fatty acid synthase; upregulated in MASH hepatocytes
#     "ACACA",     # ACC1 — rate-limiting DNL; induced by insulin/SREBP
#     "SREBF1",    # SREBP-1c — master lipogenic TF; induced in MASH
#     "DGAT2",     # diacylglycerol acyltransferase; TG synthesis; regulated
#     "PLIN2",     # lipid droplet protein; regulated by lipid load
#     "CD36",      # FA translocase; markedly upregulated in MASH hepatocytes
#     "ACSL4",     # long-chain acyl-CoA synthetase; ferroptosis sensitiser
#     "CPT1A",     # FAO rate-limiting step; suppressed in MASH
#     "PPARA",     # PPARα — FAO master regulator; most suppressed NR in MASH

#     # ── Nuclear receptors / hepatocyte TFs (8) ───────────────────────────────
#     # Expression regulated by nutritional state and disease — not constitutive.
#     "PPARG",     # PPARγ — lipogenesis; HSC quiescence
#     "NR1H4",     # FXR — bile acid/lipid homeostasis; therapeutic target
#     "HNF4A",     # central MASH network hub; reduced in advanced disease
#     "CEBPA",     # hepatocyte TF; suppressed in MASH fibrosis
#     "FOXO1",     # gluconeogenesis + insulin signalling integrator
#     "NR0B2",     # SHP — FXR-inducible metabolic gatekeeper
#     "THRB",      # THR-β — target of resmetirom (FDA approved 2024)
#     "KLF6",      # Krüppel-like factor 6 — HSC activation TF

#     # ── Oxidative stress / antioxidant / ferroptosis (8) ─────────────────────
#     # Stress-responsive — induced by ROS, not constitutively maximal.
#     "NFE2L2",    # NRF2 — stress-inducible master antioxidant TF
#     "HMOX1",     # heme oxygenase-1; strongly stress-inducible
#     "GPX4",      # ferroptosis gatekeeper; regulated by lipid peroxides
#     "SLC7A11",   # xCT — cystine import; ferroptosis; regulated
#     "NOX4",      # NADPH oxidase 4; ROS production; fibrosis driver
#     "SIRT1",     # NAD+-deacetylase; MASH-suppressed; not constitutive
#     "TXNIP",     # thioredoxin-interacting protein; oxidative stress sensor
#     "SOD2",      # mitochondrial SOD; regulated by oxidative stress

#     # ── Cell death — apoptosis / necroptosis / pyroptosis (6) ────────────────
#     # Activated/induced — not constitutively high in healthy hepatocytes.
#     "TP53",      # p53 — stress sensor; elevated in MASH
#     "CASP3",     # effector caspase — apoptosis executor; activated
#     "RIPK3",     # necroptosis kinase; elevated in MASH
#     "MLKL",      # necroptosis executor
#     "GSDMD",     # gasdermin D — pyroptosis pore; specifically induced
#     "BCL2L11",   # BIM — pro-apoptotic BH3-only; regulated

#     # ── Bile acid / CYP enzymes (4) ───────────────────────────────────────────
#     # Regulated by disease state — not constitutively maximal.
#     "CYP7A1",    # rate-limiting BA synthesis; reduced in MASH
#     "CYP2E1",    # metabolises FFAs → ROS; upregulated in MASH
#     "ABCB11",    # BSEP — bile salt export; reduced in MASH
#     "FGF19",     # ileal FXR hormone; steatosis modulator; regulated

#     # ── ER stress / UPR (4) ──────────────────────────────────────────────────
#     # Near-absent in healthy unstressed hepatocytes; induced by lipotoxicity.
#     "DDIT3",     # CHOP — pro-apoptotic; strongly stress-induced in MASH
#     "ATF3",      # hub gene MASH/ferroptosis networks (Lin 2025)
#     "XBP1",      # IRE1α substrate; UPR TF; spliced under ER stress
#     "HSPA5",     # GRP78/BiP — ER chaperone; upregulated in MASH

#     # ── Autophagy / mitophagy (3) ─────────────────────────────────────────────
#     "BECN1",     # beclin-1 — autophagy initiation; regulated
#     "SQSTM1",    # p62 — aggregates in MASH hepatocytes; regulated
#     "PINK1",     # mitophagy kinase; regulated by mitochondrial damage

#     # ── NF-κB / JAK-STAT / MAPK inflammation hubs (4) ────────────────────────
#     # Signalling molecules — activity/expression regulated by disease state.
#     "NFKB1",     # NF-κB p50 — master inflammatory TF
#     "STAT3",     # JAK-STAT; IL-6 effector; regulated
#     "MAPK8",     # JNK1 — stress kinase; insulin resistance; MASH driver
#     "JAK2",      # JAK-STAT kinase; regulated

#     # ── Insulin resistance / glucose (2) ──────────────────────────────────────
#     "INSR",      # insulin receptor — reduced signalling in MASH hepatocytes
#     "IRS1",      # insulin receptor substrate 1; regulated

#     # ── Novel scRNA-seq / spatial MASH drivers (4) ────────────────────────────
#     # Identified from recent 2024-2025 MASH multi-omics studies.
#     "EGR1",      # early growth response 1 — stress TF; MASH network hub
#     "ZFP36",     # ZFP36/TTP — mRNA stability; anti-inflammatory; regulated
#     "NAMPT",     # nicotinamide phosphoribosyltransferase; MASH-regulated
#     "GADD45B",   # stress-inducible; MASH biomarker (nomogram study 2021)

# ]

_seen = set()
_deduped = []
for g in MASH_GENES:
    if g not in _seen:
        _deduped.append(g)
        _seen.add(g)
MASH_GENES = _deduped

print(f"{len(MASH_GENES)} unique literature-curated candidate genes loaded.")
print(f"   (Single shared list — same genes tested across ALL cell types)")

# ── Load all cell-type-specific h5ad files ────────────────────────────────────
print(f"\nLoading cell-type-specific h5ad files from: {DATA_DIR}")
adata_dict = {}  # cell_type_name → adata object
cell_type_h5ad_files = sorted(DATA_DIR.glob('*.h5ad'))

for h5ad_file in cell_type_h5ad_files:
    cell_type_name = h5ad_file.stem  # filename without .h5ad extension
    # Remove '_embedded' suffix if present
    if cell_type_name.endswith('_embedded'):
        cell_type_name = cell_type_name[:-len('_embedded')]
    try:
        adata_ct = ad.read_h5ad(h5ad_file)
        adata_dict[cell_type_name] = adata_ct
        print(f"  Loaded {cell_type_name}: {adata_ct.n_obs} cells, {adata_ct.n_vars} genes")
    except Exception as e:
        print(f"  X Failed to load {cell_type_name}: {e}")

print(f"\nTotal cell types loaded: {len(adata_dict)}")

# ── Extract embeddings from all loaded adatas ──────────────────────────────────
print("\nExtracting embeddings and building metadata...")
embeddings_list = []

for cell_type_name, adata_ct in adata_dict.items():
    if EMBEDDING_KEY not in adata_ct.obsm:
        print(f"Skipping {cell_type_name}: no '{EMBEDDING_KEY}' found")
        continue
    
    # Extract embeddings
    emb_matrix = adata_ct.obsm[EMBEDDING_KEY]  # shape: (n_cells, n_dims)
    n_dims = emb_matrix.shape[1]
    emb_cols = [f'emb_{i}' for i in range(n_dims)]
    
    # Build embeddings DataFrame for this cell type
    df_emb = pd.DataFrame(emb_matrix, index=adata_ct.obs_names, columns=emb_cols)
    
    # Attach metadata columns (required: condition, donor, cell_id)
    df_emb['cell_type'] = cell_type_name  # use filename-derived cell type name
    df_emb['condition'] = adata_ct.obs[CONDITION_COL].values if CONDITION_COL in adata_ct.obs else np.nan
    df_emb['donor'] = adata_ct.obs[DONOR_COL].values if DONOR_COL in adata_ct.obs else np.nan
    df_emb['cell_id'] = adata_ct.obs_names
    
    embeddings_list.append(df_emb)
    print(f"  ✓ {cell_type_name}: {df_emb.shape[0]} cells, {len(emb_cols)} dimensions")

# Combine all embeddings
embeddings = pd.concat(embeddings_list, ignore_index=False)

# Subset to only INITIAL_LABEL and TARGET_LABEL cells
mask = embeddings['condition'].isin([INITIAL_LABEL, TARGET_LABEL])
embeddings = embeddings[mask].copy()

# Patch fix for ValueError: operands could not be broadcast together with shapes (n_cells, n_dims) and (n_dims,) during cosine similarity calculation.  
embeddings = embeddings[~embeddings.index.duplicated(keep='first')]

embeddings = embeddings.dropna(subset=['cell_type', 'condition', 'donor'])

print(f"\nEmbeddings shape (after filtering to {INITIAL_LABEL}, {TARGET_LABEL}): {embeddings.shape}")
print(f"Conditions: {embeddings['condition'].value_counts().to_dict()}")
print(f"Cell types: {embeddings['cell_type'].nunique()} unique")
print(f"Donors: {embeddings['donor'].nunique()} unique")

# Replace spaces and '/' in cell type names (required by the separation test code)
embeddings['cell_type'] = embeddings['cell_type'].str.replace(' ', '_', regex=False).str.replace('/', '_', regex=False)

# Resolve embedding feature columns
emb_meta_cols    = {'cell_type', 'condition', 'cell_id', 'donor'}
emb_feature_cols = [c for c in embeddings.columns
                    if c not in emb_meta_cols
                    and pd.api.types.is_numeric_dtype(embeddings[c])]

state_pairs = [(INITIAL_LABEL, TARGET_LABEL, f"{INITIAL_LABEL} vs {TARGET_LABEL}")]

print(f"State comparisons: {[p[2] for p in state_pairs]}")
print(f"Cell types to test ({len(PASSING_CELLTYPES)}): {PASSING_CELLTYPES}")



# Chunk 9B
import scipy.sparse as sp

# ── Prepare data for each cell type ────────────────────────────────────────────
# Build a dictionary: cell_type_name → {'adata_ctrl', 'X_counts_dense', 'gene_names'}

prepared_data = {}

for cell_type_name, adata_ct in adata_dict.items():
    # Sanitise cell type name: replace spaces and '/' with '_'
    cell_type_clean = cell_type_name.replace(' ', '_').replace('/', '_')
    
    # Subset to Healthy control cells only (ISP always starts from control)
    ctrl_mask = adata_ct.obs[CONDITION_COL] == INITIAL_LABEL
    adata_ctrl = adata_ct[ctrl_mask].copy()
    
    if adata_ctrl.n_obs == 0:
        print(f"{cell_type_name}: No control cells found — skipping.")
        continue
    
    print(f"\n  {cell_type_name}:")
    print(f"    Control cells: {adata_ctrl.n_obs:,}")
    
    X_counts = adata_ctrl.X
    print(f"    Using adata.X (normalized/log1p) — matches baseline embedding representation.")
    
    # Dense for easy indexing; use float32 to save memory
    if sp.issparse(X_counts):
        X_counts_dense = np.array(X_counts.todense(), dtype=np.float32)
    else:
        X_counts_dense = np.array(X_counts, dtype=np.float32)
    
    gene_names = adata_ctrl.var[GENE_COL].values  # gene symbol array aligned to columns
    print(f"    Gene matrix shape: {X_counts_dense.shape}")
    
    prepared_data[cell_type_clean] = {
        'adata_ctrl': adata_ctrl,
        'X_counts_dense': X_counts_dense,
        'gene_names': gene_names,
        'original_cell_type': cell_type_name
    }

print(f"\n  Prepared data for {len(prepared_data)} cell types.")

# Chunk 9C
# scGPT tokenises expression by binning normalised counts.
# Down-regulation ISP = set the gene's bin to PERTURB_BIN_VALUE (0).
# We replicate the same binning scgpt.tasks.embed_data uses internally
# so our perturbation is in the same representation space the model expects.
#
# scGPT uses n_bins=51 by default (bins 0-51). Bin 0 = not expressed.
# We zero out the raw count for the target gene, which after normalisation
# and binning maps to bin 0 — equivalent to "not expressed" / down-regulated.

N_BINS = 51  # scGPT default

def bin_expression(raw_counts_row, n_bins=N_BINS):
    """
    Replicate scGPT's expression binning for a single cell (1D array).
    Returns integer bin values aligned to the same gene axis.
    """
    total = raw_counts_row.sum()
    if total == 0:
        return np.zeros(len(raw_counts_row), dtype=np.int32)
    # Normalise to 10k (same as scGPT default)
    normed = raw_counts_row / total * 1e4
    log1p  = np.log1p(normed)
    # Bin non-zero values into n_bins equal-quantile bins
    nonzero_mask = log1p > 0
    bins = np.zeros(len(log1p), dtype=np.int32)
    if nonzero_mask.sum() > 0:
        vals = log1p[nonzero_mask]
        # scGPT uses rank-based binning: divide into n_bins equal groups
        ranks    = vals.argsort().argsort()  # rank each value
        n_nonzero = len(vals)
        bin_edges = np.linspace(0, n_nonzero, n_bins + 1)
        bin_idx   = np.searchsorted(bin_edges, ranks, side='right').clip(1, n_bins)
        bins[nonzero_mask] = bin_idx
    return bins

# Quick sanity check
_test_bins = bin_expression(X_counts_dense[0])
print(f"Bin range for first cell: {_test_bins.min()} – {_test_bins.max()}")
print(f"Non-zero bins: {((_test_bins > 0).sum())} / {len(_test_bins)}")
print(" Binning function ready.")

# Chunk 9D
def patient_aware_sample_for_isp(obs_df, donor_col,
                                  max_cells_total=MAX_ISP_CELLS_TOTAL,
                                  min_cells_per_donor=MIN_CELLS_PER_DONOR_ISP,
                                  random_state=42):
    """
    Sample cells proportionally across donors, with a minimum per-donor
    threshold. Returns (list_of_obs_index_positions, donor_report_dict).
    obs_df must be the .obs slice for the relevant cell type + condition.
    """
    rng = np.random.default_rng(random_state)
    donor_counts = obs_df[donor_col].value_counts()
    eligible     = donor_counts[donor_counts >= min_cells_per_donor].index.tolist()

    if not eligible:
        # Fall back: take all cells, up to max
        idx    = rng.choice(len(obs_df), size=min(max_cells_total, len(obs_df)),
                            replace=False)
        return idx.tolist(), {"random_fallback": len(idx)}

    # Proportional allocation
    total_eligible = donor_counts[eligible].sum()
    sampled_positions = []
    donor_report      = {}

    for donor in eligible:
        d_mask      = obs_df[donor_col] == donor
        d_positions = np.where(d_mask)[0]
        proportion  = len(d_positions) / total_eligible
        n_take      = max(1, int(round(proportion * max_cells_total)))
        n_take      = min(n_take, len(d_positions))
        chosen      = rng.choice(d_positions, size=n_take, replace=False)
        sampled_positions.extend(chosen.tolist())
        donor_report[donor] = n_take

    # Trim to budget
    if len(sampled_positions) > max_cells_total:
        sampled_positions = rng.choice(
            sampled_positions, size=max_cells_total, replace=False
        ).tolist()

    return sampled_positions, donor_report

print(" Patient-aware sampler ready.")

# Chunk 9E
from sklearn.metrics.pairwise import cosine_similarity as cos_sim_sklearn
import time

# Main ISP loop: iterate through prepared cell types
all_results = []

perturb_label = "down-regulation (bin → 0)" if PERTURB_MODE == 'down' else "up-regulation (bin → 51)"
print(f"\n Running scGPT ISP ({perturb_label})...")
print(f"   Perturbation : set gene bin → {PERTURB_BIN_VALUE} ({perturb_label})")
print(f"   Cell types   : {len(PASSING_CELLTYPES)}")
print(f"   Candidate genes: {len(MASH_GENES)}")
print(f"   Control → Target: {INITIAL_LABEL} → {TARGET_LABEL}\n")

for cell_type_clean in PASSING_CELLTYPES:
    # Normalize cell type name (replace spaces/special chars with underscores for lookup)
    cell_type_key = cell_type_clean.replace(' ', '_').replace('/', '_')
    
    # Check if this cell type is in our prepared data
    if cell_type_key not in prepared_data:
        print(f"\n{'─'*60}")
        print(f"Cell type: {cell_type_clean}")
        print(f"     Not found in prepared data — skipping.")
        continue
    
    t_start = time.time()
    print(f"\n{'─'*60}")
    print(f"Cell type: {cell_type_clean}")
    
    # Get prepared data for this cell type
    prep_data = prepared_data[cell_type_key]
    adata_ctrl = prep_data['adata_ctrl']
    X_counts_dense = prep_data['X_counts_dense']
    gene_names = prep_data['gene_names']
    
    # ── 1. State centroids from baseline embeddings ────────────────────────
    ctrl_emb_mask = ((embeddings['cell_type'] == cell_type_key) &
                     (embeddings['condition'] == INITIAL_LABEL))
    tgt_emb_mask  = ((embeddings['cell_type'] == cell_type_key) &
                     (embeddings['condition'] == TARGET_LABEL))

    if ctrl_emb_mask.sum() == 0 or tgt_emb_mask.sum() == 0:
        print(f"     Missing baseline embeddings — skipping.")
        continue

    ctrl_centroid = embeddings.loc[ctrl_emb_mask, emb_feature_cols].values.mean(axis=0)
    tgt_centroid  = embeddings.loc[tgt_emb_mask,  emb_feature_cols].values.mean(axis=0)

    # Unit-normalise centroids for cosine shift scoring
    ctrl_centroid_n = ctrl_centroid / (np.linalg.norm(ctrl_centroid) + 1e-12)
    tgt_centroid_n  = tgt_centroid  / (np.linalg.norm(tgt_centroid)  + 1e-12)

    baseline_cos = float(np.dot(ctrl_centroid_n, tgt_centroid_n))
    print(f"   Baseline cosine similarity (ctrl ↔ target): {baseline_cos:.4f}")

    # ── 2. Select control cells: all cells in adata_ctrl are already control ──
    print(f"   Control cells available: {adata_ctrl.n_obs}")

    # ── 3. Patient-aware sampling ─────────────────────────────────────────
    ct_obs = adata_ctrl.obs.copy()
    
    if DONOR_COL in ct_obs.columns:
        sampled_positions, donor_report = patient_aware_sample_for_isp(
            ct_obs, donor_col=DONOR_COL
        )
        print(f"   Sampled {len(sampled_positions)} cells | donors: {donor_report}")
    else:
        rng_fallback  = np.random.default_rng(42)
        n_take        = min(MAX_ISP_CELLS_TOTAL, adata_ctrl.n_obs)
        sampled_positions = rng_fallback.choice(adata_ctrl.n_obs, n_take,
                                                replace=False).tolist()
        print(f"     No donor column — random sample of {n_take} cells.")

    # Integer positions within adata_ctrl
    isp_positions = sampled_positions

    # Subset count matrix to ISP cells
    X_isp = X_counts_dense[isp_positions]          # (n_isp_cells, n_genes)
    n_isp_cells = len(isp_positions)
    print(f"   ISP input matrix: {X_isp.shape}")

    # ── 4. Perturb each candidate gene and collect cosine shifts ──────────
    gene_shifts = {}   # gene_symbol → array of per-cell cosine shifts

    for gene in MASH_GENES:
        # Find gene column index
        gene_col_matches = np.where(gene_names == gene)[0]
        if len(gene_col_matches) == 0:
            continue   # gene not in adata.var
        gene_idx = gene_col_matches[0]

        # Check how many ISP cells actually express this gene (count > 0)
        expressing_mask = X_isp[:, gene_idx] > 0
        n_expressing    = expressing_mask.sum()

        if n_expressing < MIN_CELLS_PER_GENE and PERTURB_MODE == 'down':
            continue   # too few cells — will be skipped in scoring

        # ── Perturb gene in raw-count space ─────────────────────────────
        # embed_data internally does: normalise → log1p → rank-based bin (0-51)
        # Down: raw count → 0           guarantees bin 0  (gene silenced)
        # Up:   raw count → 10× cell max guarantees bin 51 (gene dominates rank)
        X_perturbed = X_isp.copy()
        if PERTURB_MODE == 'down':
            X_perturbed[:, gene_idx] = 0.0
        else:  # 'up'
            # Per-cell row max: each cell's highest count × 10 forces this gene
            # to rank first after normalisation → bin 51
            cell_maxes = X_isp.max(axis=1)          # shape: (n_cells,)
            X_perturbed[:, gene_idx] = cell_maxes * 10.0

        # ── Re-embed perturbed cells through scGPT ────────────────────────
        # Build a temporary AnnData with only ISP cells + perturbed counts
        adata_perturbed = ad.AnnData(
            X=X_perturbed,
            obs=adata_ctrl.obs.iloc[isp_positions].copy(),
            var=adata_ctrl.var.copy()
        )
        # scGPT embed_data expects the raw counts in .X
        adata_perturbed.layers['counts'] = X_perturbed.copy()

        try:
            adata_emb = scg.tasks.embed_data(
                adata_perturbed,
                SCGPT_MODEL_DIR,
                gene_col=GENE_COL,
                batch_size=64,
                return_new_adata=False,
            )
            perturbed_embs = adata_emb.obsm['X_scGPT']  # (n_cells, n_dims)
        except Exception as e:
            print(f"     embed_data failed for {gene}: {e}")
            traceback.print_exc()
            continue

        # ── Cosine shift: per-cell distance toward target centroid ────────
        # shift_i = cos(perturbed_i, target_centroid) - cos(unperturbed_i, target_centroid)
        # Positive shift = perturbation moved cell toward target state
        
        # Align: only score cells present in both
        shared_idx = [
            i for i, p in enumerate(isp_positions)
            if adata_ctrl.obs_names[p] in embeddings.index
        ]
        if len(shared_idx) < MIN_CELLS_PER_GENE:
            continue

        # Extract embeddings only for shared cells, in same order
        shared_cell_names = [adata_ctrl.obs_names[p] for p in [isp_positions[i] for i in shared_idx]]
        X_unperturbed_emb = embeddings.loc[shared_cell_names, emb_feature_cols].values
        
        perturbed_embs_shared   = perturbed_embs[shared_idx]
        unperturbed_embs_shared = X_unperturbed_emb

        tgt_vec = tgt_centroid_n.reshape(1, -1)
        cos_perturbed   = cos_sim_sklearn(perturbed_embs_shared,   tgt_vec).flatten()
        cos_unperturbed = cos_sim_sklearn(unperturbed_embs_shared, tgt_vec).flatten()

        shifts = cos_perturbed - cos_unperturbed   # positive = toward target
        gene_shifts[gene] = shifts

    print(f"   Genes with sufficient data: {len(gene_shifts)}/{len(MASH_GENES)}")
    
    # ── 5. Score genes for this cell type  [Geneformer-equivalent framework] ──────
    if not gene_shifts:
        print(f"     No gene data for {cell_type_clean} — skipping.")
        continue

    # ── 4B. Cache raw per-cell shifts for later re-analysis ────────────────
    # Persist the raw shift arrays so the statistical methodology (null
    # construction, one-sided vs two-sided, resampling vs full-pool) can be
    # revised later WITHOUT re-running embed_data. embed_data is the
    # expensive step (GPU inference per gene, per cell type); everything
    # after this point is cheap and should never require a rerun again.
    import json

    RAW_SHIFTS_DIR = OUTPUT_DIR / "raw_shifts_cache"
    os.makedirs(RAW_SHIFTS_DIR, exist_ok=True)

    cache_path = RAW_SHIFTS_DIR / f"{cell_type_key}.npz"
    np.savez_compressed(cache_path, **{g: v for g, v in gene_shifts.items()})

    cache_meta_path = RAW_SHIFTS_DIR / f"{cell_type_key}_meta.json"
    with open(cache_meta_path, 'w') as f:
        json.dump({
            'cell_type_key':      cell_type_key,
            'original_cell_type':   prep_data['original_cell_type'],
            'data_type':            DATA_TYPE,
            'initial_label':        INITIAL_LABEL,
            'target_label':         TARGET_LABEL,
            'perturb_mode':         PERTURB_MODE,
            'n_candidate_genes':    len(MASH_GENES),
            'n_genes_with_data':    len(gene_shifts),
            'min_cells_per_gene':   MIN_CELLS_PER_GENE,
            'effect_size_threshold': EFFECT_SIZE_THRESHOLD,
        }, f, indent=2)

    print(f"   Cached raw shifts: {cache_path.name} ({len(gene_shifts)} genes)")

    from scipy.stats import ranksums
    from statsmodels.stats.multitest import multipletests

    # Split gene_shifts into candidate genes vs. all other genes scored this
    # cell type.  "Other genes" = any gene that was scored but is NOT in
    # MASH_GENES — used as the genome-wide null, exactly as Geneformer does.
    # Because your pipeline only perturbs MASH_GENES, there is no separate
    # "other" pool; we therefore fall back to a leave-one-out candidate pool
    # (same genes, exclude the gene being tested), and apply the Geneformer
    # floor-clip rule: if the null median < 0, clip to 0.

    rng_score = np.random.default_rng(42)

    rows = []
    for gene, shifts_a in gene_shifts.items():
        shifts_a = shifts_a[np.isfinite(shifts_a)]
        if len(shifts_a) < MIN_CELLS_PER_GENE:
            continue

        n_a          = len(shifts_a)
        median_shift = float(np.nanmedian(shifts_a))
        mean_shift   = float(np.nanmean(shifts_a))
        std_shift    = float(np.nanstd(shifts_a))

        # Build null: all other candidate genes (leave-one-out), mirroring
        # Geneformer's "other_gene_shifts" pool.
        other_shifts = np.concatenate([
            v for k, v in gene_shifts.items() if k != gene
        ])
        other_shifts = other_shifts[np.isfinite(other_shifts)]

        # Null clip — symmetric to perturbation direction:
        # Down: floor-clip at 0 — prevents a negatively-biased null from
        #        artificially inflating significance of downward-shifting genes.
        # Up:   ceiling-clip at 0 — prevents a positively-biased null from
        #        artificially suppressing significance of upward-shifting genes.
        if PERTURB_MODE == 'down':
            if len(other_shifts) > 0 and np.median(other_shifts) < 0:
                other_shifts = np.maximum(other_shifts, 0.0)
        else:  # 'up'
            if len(other_shifts) > 0 and np.median(other_shifts) > 0:
                other_shifts = np.minimum(other_shifts, 0.0)

        # Use the full combined null pool — no down-sampling to n_a.
        # (scISP paper explicitly rejected size-matched resampling in favor
        # of the full combined pool for stability/reproducibility; ranksums
        # does not require equal sample sizes.)
        if len(other_shifts) > 0:
            sample_b = other_shifts
        else:
            sample_b = np.zeros(n_a)

        # One-sided test: is shifts_a significantly GREATER than the null?
        # shifts_a = cos(perturbed, target) - cos(unperturbed, target), so
        # positive values always mean "moved toward target" regardless of
        # PERTURB_MODE — matching the paper's "sample A higher than sample B"
        # success criterion.
        try:
            _, pval = ranksums(shifts_a, sample_b) #, alternative='greater') - changed to match Geneformer
        except Exception:
            pval = 1.0
        if not np.isfinite(pval):
            pval = 1.0

        rows.append({
            'gene_symbol':         gene,
            'cell_type':           cell_type_key,
            'target_state':        TARGET_LABEL,
            'median_cosine_shift': median_shift,
            'mean_cosine_shift':   mean_shift,
            'std_cosine_shift':    std_shift,
            'n_cells':             n_a,
            'pval_raw':            float(pval),
            'baseline_cos_sim':    baseline_cos,
        })

    if not rows:
        continue

    ct_df = pd.DataFrame(rows)
    ct_df['pval_raw']  = ct_df['pval_raw'].fillna(1.0)

    # FDR correction within cell type (same as Geneformer)
    _, pval_adj, _, _ = multipletests(ct_df['pval_raw'].values,
                                    alpha=0.05, method='fdr_bh')
    ct_df['pval_adj'] = pval_adj

    # Significance criteria — Geneformer delete-mode equivalent:
    # FDR < 0.05  AND  |median shift| > threshold  AND  n_cells >= minimum
    ct_df['significant'] = (
        (ct_df['pval_adj'] < 0.05) &
        (ct_df['median_cosine_shift'].abs() > EFFECT_SIZE_THRESHOLD) &
        (ct_df['n_cells'] >= MIN_CELLS_PER_GENE)
    )
    ct_df.loc[ct_df['n_cells'] < MIN_CELLS_PER_GENE, 'significant'] = False

    # For down-regulation (delete-mode): sort ascending (negative shifts first)
    # For up-regulation (add-mode): sort descending (positive shifts first)
    if PERTURB_MODE == 'down':
        ct_df = ct_df.sort_values('median_cosine_shift', ascending=True).reset_index(drop=True)
    elif PERTURB_MODE == 'up':
        ct_df = ct_df.sort_values('median_cosine_shift', ascending=False).reset_index(drop=True)

    all_results.append(ct_df)

    n_sig = ct_df['significant'].sum()
    elapsed = (time.time() - t_start) / 60
    print(f"    Done ({elapsed:.1f} min) | {n_sig} significant genes")
    if n_sig > 0:
        print(ct_df[ct_df['significant']][
            ['gene_symbol', 'median_cosine_shift', 'pval_adj', 'n_cells']
        ].head(10).to_string(index=False))

    # Save per-cell-type results
    safe_ct  = cell_type_clean.replace(' ', '_').replace('/', '_')
    ct_path  = os.path.join(ISP_OUTPUT_DIR, f"isp_{safe_ct}.csv")
    ct_df.to_csv(ct_path, index=False)

print("\n scGPT ISP complete.")

# Chunk 9F
if all_results:
    results_df = pd.concat(all_results, ignore_index=True)
    results_df = results_df.dropna(subset=['median_cosine_shift'])

    full_path = os.path.join(STATS_OUTPUT_DIR, 'scgpt_isp_results_full.csv')
    results_df.to_csv(full_path, index=False)
    print(f"Saved full results: {full_path}")
    print(f"Total rows: {len(results_df)}")
    print(f"Significant hits: {results_df['significant'].sum()}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {INITIAL_LABEL} → {TARGET_LABEL}")
    print(f"{'='*60}")

    for ct in PASSING_CELLTYPES:
        ct_clean = ct.replace(' ', '_').replace('/', '_')
        ct_sig = results_df[
            (results_df['cell_type'] == ct_clean) & results_df['significant']
        ].sort_values('median_cosine_shift', ascending=False)

        print(f"\n  {ct} ({len(ct_sig)} significant):")
        if len(ct_sig) > 0:
            print(ct_sig[['gene_symbol', 'median_cosine_shift',
                           'pval_adj', 'n_cells']].head(10).to_string(index=False))
        else:
            print("    No significant hits.")
else:
    print("No ISP results. Check earlier steps.")