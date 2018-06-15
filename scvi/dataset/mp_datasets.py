'''
Datasets from the paper:
"Massively parallel digital transcriptional profiling of single cells"
'''
import os
import shutil
from subprocess import call

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

from .dataset import GeneExpressionDataset

# In this paper, they prioritize label n. 2 for better visualization (D4+/CD45RA+/CD25- Naive T)
key_svmlight = ["CD34+", "CD56+ NK", "CD4+/CD45RA+/CD25- Naive T", "CD4+/CD25 T Reg", "CD8+/CD45RA+ Naive Cytotoxic",
                "CD4+/CD45RO+ Memory", "CD8+ Cytotoxic T", "CD19+ B", "CD4+ T Helper2", "CD14+ Monocyte", "Dendritic"]

key_color_order = ["CD19+ B", "CD14+ Monocyte", "Dendritic", "CD56+ NK", "CD34+", "CD4+/CD25 T Reg",
                   "CD4+/CD45RA+/CD25- Naive T", "CD4+/CD45RO+ Memory", "CD4+ T Helper2",
                   "CD8+/CD45RA+ Naive Cytotoxic",
                   "CD8+ Cytotoxic T"]

colors = ["#1C86EE",  # 1c86ee dodgerblue2
          "#008b00",  # green 4
          "#6A3D9A",  # purple
          "grey",
          "#8B5A2B",  # tan4
          "yellow",
          "#FF7F00",  # orange
          "black",
          "#FB9A99",  # pink
          "#ba55d3",  # orchid
          "red"]

key_names_color = dict(zip(key_color_order, colors))

index_to_color = dict([(i, (k, key_names_color[k])) for i, k in enumerate(key_svmlight)])


def run_r_preprocessing():
    # Downloading raw datasets and saving them in data/mp/
    urls = ["http://s3-us-west-2.amazonaws.com/10x.files/samples/cell/pbmc68k_rds/pbmc68k_data.rds",
            "http://s3-us-west-2.amazonaws.com/10x.files/samples/cell/pbmc68k_rds/all_pure_pbmc_data.rds",
            "http://s3-us-west-2.amazonaws.com/10x.files/samples/cell/pbmc68k_rds/all_pure_select_11types.rds"]
    for url in urls:
        GeneExpressionDataset._download(url=url, save_path="data/mp/", download_name=url.split('/')[-1])

    # Then run the prepcrocessing R scripts. There are R dependencies to solve (library(svd), ect...)
    # Run the paper analysis to infer labels to the donor dataset.
    # Saves in .svmlight (sparse) format the purified and donor datasets.
    # Sometimes a memory error.
    call(["git", "clone", "https://github.com/Edouard360/single-cell-3prime-paper.git"])
    call(["Rscript", "single-cell-3prime-paper/pbmc68k_analysis/main_python_R.R"])

    # Remove the cloned repository containind the preprocessing scripts.
    shutil.rmtree('single-cell-3prime-paper/')
    print("Preprocessing done")


class PurePBMC(GeneExpressionDataset):
    def __init__(self):
        path = "data/mp/pure_full.svmlight"
        if not os.path.exists(path):
            run_r_preprocessing()
        sparse_matrix, labels = load_svmlight_file(path)
        labels = labels - 1

        print("Purified PBMC dataset loaded with shape : ", sparse_matrix.shape)

        # When saving in .svmlight format, the last 5 genes (with only 0s) are not saved
        # As well as the first gene (with only 0s too).
        gene_names = pd.read_csv("data/mp/gene_names.csv").values.astype(np.str).ravel()
        gene_symbols = pd.read_csv("data/mp/gene_symbols.csv").values.astype(np.str).ravel()
        gene_names = ["ENSG00000" + suffix for suffix in gene_names[1:-5]]
        self.gene_symbols = gene_symbols[1:-5]

        # Consider bead enriched subpopulations as individual batches
        super(PurePBMC, self).__init__(
            *GeneExpressionDataset.get_attributes_from_list(
                [sparse_matrix[labels == i] for i in range(int(labels[-1] + 1))],
                list_labels=[labels[labels == i] for i in range(int(labels[-1] + 1))]),
            gene_names=gene_names)


class DonorPBMC(GeneExpressionDataset):
    def __init__(self):
        path = "data/mp/68k_assignments.svmlight"
        if not os.path.exists(path):
            run_r_preprocessing()
        sparse_matrix, labels = load_svmlight_file(path)
        labels = labels - 1
        # Labels here are not ground truth. They are inferred from correlation scores (cf. paper).

        print("Donor PBMC dataset loaded with shape : ", sparse_matrix.shape)

        # When saving in .svmlight format, the last 5 genes (with only 0s) are not saved
        # As well as the first gene (with only 0s too).
        gene_names = pd.read_csv("data/mp/gene_names.csv").values.astype(np.str).ravel()
        gene_symbols = pd.read_csv("data/mp/gene_symbols.csv").values.astype(np.str).ravel()
        gene_names = ["ENSG00000" + suffix for suffix in gene_names[1:-5]]
        self.gene_symbols = gene_symbols[1:-5]

        # TODO: include more information about the individual predictions for according to the
        # TODO: highest correlation method presented in the paper. (For comparison against them).
        super(DonorPBMC, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                sparse_matrix,
                labels=labels),
            gene_names=gene_names)
