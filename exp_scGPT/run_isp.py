# Python Script to run ISP using scGPT

# Chunk 9A
import scanpy as sc
import anndata as ad
from scipy import sparse
from scipy.stats import median_abs_deviation
import os
import numpy as np
import pandas as pd
import torch
import scgpt as scg
from pathlib import Path
import traceback
import warnings
warnings.filterwarnings('ignore')

# ── scGPT ISP config ───────────────────────────────────────────────────────────
SCGPT_MODEL_DIR     = Path("/mnt/hd1/home/ankitray/scFM/scGPT_CP")   # same model dir you used for embeddings
GENE_COL            = "Gene Symbol"             # adata.var column with gene symbols
CONTROL_LABEL       = "Healthy"
TARGET_LABEL        = "MASH"
CONDITION_COL       = "broad_condition"
CELL_TYPE_COL       = "typist_liver_majority_voting"
DONOR_COL           = "patient_id"

EMBEDDING_KEY   = 'X_scGPT'          # adata.obsm key
MIN_CELLS_PER_STATE = 30
FOCUS_CELLTYPES = []                  # leave empty to test all cell types
ALT_STATES      = []                  # no alt states
OUTPUT_DIR      = '/mnt/hd1/home/ankitray/scFM/results/scGPT/separation_test_output_celltype_liver'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ISP_CELLS_TOTAL     = 500    # control cells per cell type fed into ISP
MIN_CELLS_PER_DONOR_ISP = 5      # donors with fewer cells are excluded
MIN_CELLS_PER_GENE      = 20     # min cells a gene must appear in to be scored
EFFECT_SIZE_THRESHOLD   = 2.5e-4 # minimum |median cosine shift| to be considered
PERTURB_BIN_VALUE       = 0      # bin to set for down-regulation (0 = silenced)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

ISP_OUTPUT_DIR   = OUTPUT_DIR / "scgpt_isp_output"
STATS_OUTPUT_DIR = OUTPUT_DIR / "scgpt_isp_stats"
os.makedirs(ISP_OUTPUT_DIR,   exist_ok=True)
os.makedirs(STATS_OUTPUT_DIR, exist_ok=True)

# Set Base directory to the grandparent location of this script
BASE = Path('/mnt/hd1/home/ankitray/scFM/')

# Set working directory to the location of this script 
data_dir = BASE / 'data/scGPT_data'
# out_path = data_dir / "scgpt_immune_merged_may_11_26.h5ad"
out_path = data_dir / "scgpt_immune_merged_may_13_26_5000HVG.h5ad"
adata = ad.read_h5ad(out_path)

# Deduplicate while preserving order
candidate_genes_path = "/mnt/hd1/home/ankitray/scFM/data/candidate_gene.csv"
MASH_GENES = pd.read_csv(candidate_genes_path, header=None)[0].tolist()

_seen = set()
_deduped = []
for g in MASH_GENES:
    if g not in _seen:
        _deduped.append(g)
        _seen.add(g)
MASH_GENES = _deduped

print(f"{len(MASH_GENES)} unique literature-curated MASH/liver candidate genes loaded.")
print(f"   (Single shared list — same genes tested across ALL cell types)")

# Make Embeddings
# Extract scGPT embedding matrix
emb_matrix = adata.obsm[EMBEDDING_KEY]  # shape: (n_cells, n_dims)
n_dims = emb_matrix.shape[1]
emb_cols = [f'emb_{i}' for i in range(n_dims)]

# Build embeddings DataFrame expected by the separation test code
embeddings = pd.DataFrame(emb_matrix, index=adata.obs_names, columns=emb_cols)

# Attach required metadata columns
embeddings['cell_type'] = adata.obs[CELL_TYPE_COL].values
embeddings['condition'] = adata.obs[CONDITION_COL].values
embeddings['donor']     = adata.obs[DONOR_COL].values
embeddings['cell_id']   = adata.obs_names

# Subset to only Healthy and MASH cells
mask = embeddings['condition'].isin([CONTROL_LABEL, TARGET_LABEL])
embeddings = embeddings[mask].copy()
embeddings = embeddings.dropna(subset=['cell_type', 'condition', 'donor'])

print(f"Embeddings shape: {embeddings.shape}")
print(f"Conditions: {embeddings['condition'].value_counts().to_dict()}")
print(f"Cell types: {embeddings['cell_type'].nunique()} unique")
print(f"Donors: {embeddings['donor'].nunique()} unique")

# Replace '/' in cell type names (required by the separation test code)
embeddings['cell_type'] = embeddings['cell_type'].str.replace('/', '_', regex=False)

if FOCUS_CELLTYPES:
    FOCUS_CELLTYPES = [ct.replace('/', '_') for ct in FOCUS_CELLTYPES]

# Resolve which cell types and state pairs to test
emb_meta_cols    = {'cell_type', 'condition', 'cell_id', 'donor'}
emb_feature_cols = [c for c in embeddings.columns
                    if c not in emb_meta_cols
                    and pd.api.types.is_numeric_dtype(embeddings[c])]

cell_types_to_test = (FOCUS_CELLTYPES if FOCUS_CELLTYPES
                      else embeddings['cell_type'].unique().tolist())
cell_types_to_test = [ct for ct in cell_types_to_test
                      if ct in embeddings['cell_type'].values]

_donor_col_in_emb = None
for _dc in ['donor', 'Donor', 'donor_id', 'patient_id', 'sample_id']:
    if _dc in embeddings.columns:
        _donor_col_in_emb = _dc
        print(f'Using donor column: "{_donor_col_in_emb}"')
        break

state_pairs = [(CONTROL_LABEL, TARGET_LABEL, f"{CONTROL_LABEL} vs {TARGET_LABEL}")]

print(f"State comparisons: {[p[2] for p in state_pairs]}")
print(f"Cell types to test ({len(cell_types_to_test)}): {cell_types_to_test}")

# Passing cell types from your separation test
passing_celltypes = [
    'CD16- NK cells', 'MAIT cells', 'pDC', 'Tem_Trm cytotoxic T cells',
    'Classical monocytes', 'Plasma cells', 'CD16+ NK cells', 'DC1',
    'Tcm_Naive helper T cells', 'DC2', 'Tem_Effector helper T cells',
    'Mast cells', 'Tem_Temra cytotoxic T cells', 'Non-classical monocytes',
    'CRTAM+ gamma-delta T cells', 'Regulatory T cells', 'Memory B cells',
    'Naive B cells', 'NK cells', 'Tcm_Naive cytotoxic T cells'
]

# 5000 HVG Passing Cell Types
['CD16- NK cells', 'MAIT cells', 'pDC', 'Tem_Trm cytotoxic T cells', 
'Classical monocytes', 'Plasma cells', 'CD16+ NK cells', 'DC1', 
'Tcm_Naive helper T cells', 'DC2', 'Tem_Effector helper T cells', 
'Mast cells', 'Tem_Temra cytotoxic T cells', 'Non-classical monocytes', 
'CRTAM+ gamma-delta T cells', 'Regulatory T cells', 'Memory B cells', 
'Naive B cells', 'NK cells', 'Tcm_Naive cytotoxic T cells']

print(f" Config set. {len(passing_celltypes)} passing cell types.")



# Chunk 9B
import scipy.sparse as sp

# Sanitise cell type names to match separation test (/ → _)
adata.obs['cell_type_isp'] = (
    adata.obs[CELL_TYPE_COL].str.replace('/', '_', regex=False)
)

# Subset to Healthy control cells only (ISP always starts from control)
ctrl_mask = adata.obs[CONDITION_COL] == CONTROL_LABEL
adata_ctrl = adata[ctrl_mask].copy()
print(f"Control cells available: {adata_ctrl.n_obs:,}")
print(f"Conditions check: {adata_ctrl.obs[CONDITION_COL].unique().tolist()}")

# Confirm raw counts are in layers['counts']
# scGPT binning needs raw integer counts
if 'counts' in adata_ctrl.layers:
    X_counts = adata_ctrl.layers['counts']
    print("Using layers['counts'] for binning.")
elif adata_ctrl.raw is not None:
    X_counts = adata_ctrl.raw[:, adata_ctrl.var_names].X
    print("Using adata.raw for binning.")
else:
    X_counts = adata_ctrl.X
    print("  Using adata.X — verify these are raw counts.")

# Dense for easy indexing; use float32 to save memory
if sp.issparse(X_counts):
    X_counts_dense = np.array(X_counts.todense(), dtype=np.float32)
else:
    X_counts_dense = np.array(X_counts, dtype=np.float32)

gene_names = adata_ctrl.var[GENE_COL].values  # gene symbol array aligned to columns
print(f"Gene matrix shape: {X_counts_dense.shape}")

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

# embeddings DataFrame and emb_feature_cols must be defined
# (from your separation test chunks — they hold the baseline scGPT embeddings)

all_results = []

print(f"\n Running scGPT ISP (down-regulation by zeroing expression bin)...")
print(f"   Perturbation : set gene bin → {PERTURB_BIN_VALUE} (down-regulation)")
print(f"   Cell types   : {len(passing_celltypes)}")
print(f"   Candidate genes: {len(MASH_GENES)}")
print(f"   Control → Target: {CONTROL_LABEL} → {TARGET_LABEL}\n")

for cell_type in passing_celltypes:
    # TRIAL = 1
    t_start = time.time()
    print(f"\n{'─'*60}")
    print(f"Cell type: {cell_type}")

    # ── 1. State centroids from baseline embeddings ────────────────────────
    ctrl_emb_mask = ((embeddings['cell_type'] == cell_type) &
                     (embeddings['condition'] == CONTROL_LABEL))
    tgt_emb_mask  = ((embeddings['cell_type'] == cell_type) &
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

    # ── 2. Select control cells of this cell type ─────────────────────────
    ct_ctrl_mask = (
        (adata_ctrl.obs['cell_type_isp'] == cell_type)
    )
    ct_obs = adata_ctrl.obs[ct_ctrl_mask]

    if len(ct_obs) == 0:
        print(f"     No control cells found in adata — skipping.")
        continue

    print(f"   Control cells available: {len(ct_obs)}")

    # ── 3. Patient-aware sampling ─────────────────────────────────────────
    if DONOR_COL in ct_obs.columns:
        sampled_positions, donor_report = patient_aware_sample_for_isp(
            ct_obs, donor_col=DONOR_COL
        )
        print(f"   Sampled {len(sampled_positions)} cells | donors: {donor_report}")
    else:
        rng_fallback  = np.random.default_rng(42)
        n_take        = min(MAX_ISP_CELLS_TOTAL, len(ct_obs))
        sampled_positions = rng_fallback.choice(len(ct_obs), n_take,
                                                replace=False).tolist()
        print(f"     No donor column — random sample of {n_take} cells.")

    # Integer positions within adata_ctrl (not the full adata)
    ct_ctrl_positions = np.where(ct_ctrl_mask)[0]
    isp_positions     = [ct_ctrl_positions[i] for i in sampled_positions]

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

        if n_expressing < MIN_CELLS_PER_GENE:
            continue   # too few cells — will be skipped in scoring

        # ── Perturb: set this gene's raw count to 0 (→ bin 0) ────────────
        X_perturbed = X_isp.copy()
        X_perturbed[:, gene_idx] = PERTURB_BIN_VALUE  # zero out = down-regulate

        # ── Re-embed perturbed cells through scGPT ────────────────────────
        # Build a temporary AnnData with only ISP cells + perturbed counts
        import anndata as ad
        adata_perturbed = ad.AnnData(
            X=X_perturbed,
            obs=adata_ctrl.obs.iloc[isp_positions].copy(),
            var=adata_ctrl.var.copy()
        )
        # scGPT embed_data expects the raw counts in .X
        adata_perturbed.layers['counts'] = X_perturbed.copy()

        ######################## NA CHECK #############################
        print('#'*20)
        print(f"NaNs in gene col: {adata_perturbed.var[GENE_COL].isna().sum()}")
        print(f"dtype: {adata_perturbed.var[GENE_COL].dtype}")

        # See what the NaN rows look like
        print(adata_perturbed.var[adata_perturbed.var[GENE_COL].isna()].head(10))
        print('#'*10)
        print(f"Non NaNs in gene col: {adata_perturbed.var[GENE_COL].notna().sum()}")
        print(f"dtype: {adata_perturbed.var[GENE_COL].dtype}")

        # See what the NaN rows look like
        print(adata_perturbed.var[adata_perturbed.var[GENE_COL].notna()].head(10))
        print('#'*20)
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
        # Positive shift = perturbation moved cell toward MASH state
        X_unperturbed_emb = embeddings.loc[
            [adata_ctrl.obs_names[p] for p in isp_positions
             if adata_ctrl.obs_names[p] in embeddings.index],
            emb_feature_cols
        ].values

        # Align: only score cells present in both
        shared_idx = [
            i for i, p in enumerate(isp_positions)
            if adata_ctrl.obs_names[p] in embeddings.index
        ]
        if len(shared_idx) < MIN_CELLS_PER_GENE:
            continue

        perturbed_embs_shared   = perturbed_embs[shared_idx]
        unperturbed_embs_shared = X_unperturbed_emb

        tgt_vec = tgt_centroid_n.reshape(1, -1)
        cos_perturbed   = cos_sim_sklearn(perturbed_embs_shared,   tgt_vec).flatten()
        cos_unperturbed = cos_sim_sklearn(unperturbed_embs_shared, tgt_vec).flatten()

        shifts = cos_perturbed - cos_unperturbed   # positive = toward MASH
        gene_shifts[gene] = shifts

    print(f"   Genes with sufficient data: {len(gene_shifts)}/{len(MASH_GENES)}")

    # ── 5. Score genes for this cell type ─────────────────────────────────
    if not gene_shifts:
        print(f"     No gene data for {cell_type} — skipping.")
        continue

    # Random baseline: pool all shifts across all genes to get null distribution
    all_shifts_pool  = np.concatenate(list(gene_shifts.values()))
    all_shifts_pool  = all_shifts_pool[np.isfinite(all_shifts_pool)]
    rng_score        = np.random.default_rng(42)

    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    rows = []
    for gene, shifts in gene_shifts.items():
        shifts = shifts[np.isfinite(shifts)]
        if len(shifts) < MIN_CELLS_PER_GENE:
            continue

        median_shift = float(np.median(shifts))
        mean_shift   = float(np.mean(shifts))
        n_cells      = len(shifts)

        # Random baseline sample (same n as this gene)
        null_sample = rng_score.choice(all_shifts_pool, size=n_cells, replace=True)
        _, pval = mannwhitneyu(shifts, null_sample, alternative='two-sided')

        rows.append({
            'gene_symbol':       gene,
            'cell_type':         cell_type,
            'target_state':      TARGET_LABEL,
            'median_cosine_shift': median_shift,
            'mean_cosine_shift':  mean_shift,
            'n_cells':            n_cells,
            'pval_raw':           pval,
        })

    if not rows:
        continue

    ct_df = pd.DataFrame(rows)

    # FDR correction within cell type
    _, pval_adj, _, _ = multipletests(ct_df['pval_raw'].values, method='fdr_bh')
    ct_df['pval_adj'] = pval_adj

    # Significant = FDR < 0.05 AND |median shift| > threshold
    ct_df['significant'] = (
        (ct_df['pval_adj'] < 0.05) &
        (ct_df['median_cosine_shift'].abs() > EFFECT_SIZE_THRESHOLD)
    )

    # Sort: top hits first (largest positive shift toward MASH = best hits)
    ct_df = ct_df.sort_values('median_cosine_shift', ascending=False)

    all_results.append(ct_df)

    n_sig = ct_df['significant'].sum()
    elapsed = (time.time() - t_start) / 60
    print(f"    Done ({elapsed:.1f} min) | {n_sig} significant genes")
    if n_sig > 0:
        print(ct_df[ct_df['significant']][
            ['gene_symbol', 'median_cosine_shift', 'pval_adj', 'n_cells']
        ].head(10).to_string(index=False))

    # Save per-cell-type results
    safe_ct  = cell_type.replace(' ', '_').replace('/', '_')
    ct_path  = os.path.join(ISP_OUTPUT_DIR, f"isp_{safe_ct}.csv")
    ct_df.to_csv(ct_path, index=False)

    # if TRIAL == 1:
    #     break

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
    print(f"SUMMARY: {CONTROL_LABEL} → {TARGET_LABEL}")
    print(f"{'='*60}")

    for ct in passing_celltypes:
        ct_sig = results_df[
            (results_df['cell_type'] == ct) & results_df['significant']
        ].sort_values('median_cosine_shift', ascending=False)

        print(f"\n  {ct} ({len(ct_sig)} significant):")
        if len(ct_sig) > 0:
            print(ct_sig[['gene_symbol', 'median_cosine_shift',
                           'pval_adj', 'n_cells']].head(10).to_string(index=False))
        else:
            print("    No significant hits.")
else:
    print("No ISP results. Check earlier steps.")