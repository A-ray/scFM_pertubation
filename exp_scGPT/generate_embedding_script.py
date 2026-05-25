from pathlib import Path
import warnings

import scanpy as sc
import scib
import numpy as np
import sys
import anndata as ad 
import scgpt as scg
import matplotlib.pyplot as plt

plt.style.context('default')
warnings.simplefilter("ignore", ResourceWarning)

model_dir = Path("/mnt/hd1/home/ankitray/scFM/scFM_pertubation/scGPT_CP")

# Set Base directory to the location of this script
BASE = Path('/mnt/hd1/home/ankitray/scFM/')

# Set working directory to the location of this script 
# data_dir = BASE / 'h5s_common_directory'
liver_data_dir = BASE / 'data/liver_cell_type_adata'

# Read in each h5ad file with the Anndata object variable name being it's file name without the extension and store in dictionary
adata_dict = {}

for h5ad_file in liver_data_dir.glob('*.h5ad'):
    adata_name = h5ad_file.stem  # Get file name without extension
    adata_dict[adata_name] = ad.read_h5ad(h5ad_file)  # Read h5ad file and store in dictionary


gene_col = "Gene Symbol"
cell_type_key = "typist_liver_majority_voting"
batch_key = "GSE"
N_HVG = 5000

for adata_name, adata in adata_dict.items():
    adata_dict[adata_name].var[gene_col] = adata.var.index.values

for adata_name, adata in adata_dict.items():
    print(f"Processing {adata_name}...")
    # Could be flavor seurat_v3 with raw counts layer or cell_ranger
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3", layer="raw_counts")
    adata = adata[:, adata.var['highly_variable']]
    print(adata.shape)
    adata_dict[adata_name] = adata  # Update the dictionary with the filtered Anndata object

import torch
# Check if Torch Cuda is available
torch.cuda.is_available()

model_dir = Path("/mnt/hd1/home/ankitray/scFM/scFM_pertubation/scGPT_CP")
for adata_name, adata in adata_dict.items():
    print(f"Generating embeddings for {adata_name}...")
    embed_adata = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col=gene_col,
        batch_size=64,
    )
    adata_dict[adata_name] = embed_adata  # Update the dictionary with the embedded Anndata object


# Set Base directory to the grandparent location of this script
BASE = Path('/mnt/hd1/home/ankitray/scFM/')

out_path = BASE / "data/liver_cell_type_adata_embedded/"
# Make directory if it doesn't exist
out_path.mkdir(parents=True, exist_ok=True)

for adata_name, adata in adata_dict.items():
    print(f"Saving {adata_name}...")
    adata.write_h5ad(out_path / f"{adata_name}_embedded.h5ad")

print("All done!")