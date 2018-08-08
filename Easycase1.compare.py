use_cuda = True
from scvi.harmonization.utils_chenling import get_matrix_from_dir
from scvi.harmonization.benchmark import assign_label
import numpy as np
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.harmonization.utils_chenling import eval_latent, run_model
import sys

model_type = str(sys.argv[1])
option = str(sys.argv[2])
plotname = 'Easy1'+option
print(model_type)

count, geneid, cellid = get_matrix_from_dir('pbmc8k')
geneid = geneid[:, 1]
count = count.T.tocsr()
seurat = np.genfromtxt('../pbmc8k/pbmc8k.seurat.labels', dtype='str', delimiter=',')
cellid = np.asarray([x.split('-')[0] for x in cellid])
labels_map = [0, 2, 4, 4, 0, 3, 3, 1, 5, 6]
cell_type = ["CD4+ T Helper2", "CD56+ NK", "CD14+ Monocyte", "CD19+ B", "CD8+ Cytotoxic T", "FCGR3A Monocyte",
             "dendritic"]
dataset1 = assign_label(cellid, geneid, labels_map, count, cell_type, seurat)

count, geneid, cellid = get_matrix_from_dir('cite')
count = count.T.tocsr()
seurat = np.genfromtxt('../cite/cite.seurat.labels', dtype='str', delimiter=',')
cellid = np.asarray([x.split('-')[0] for x in cellid])
labels_map = [0, 0, 1, 2, 3, 4, 5, 6]
labels = seurat[1:, 4]
cell_type = ["CD4+ T Helper2", "CD56+ NK", "CD14+ Monocyte", "CD19+ B", "CD8+ Cytotoxic T", "FCGR3A Monocyte", "na"]
dataset2 = assign_label(cellid, geneid, labels_map, count, cell_type, seurat)
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
rmCellTypes = {'na', 'dendritic'}
newCellType = [k for i, k in enumerate(gene_dataset.cell_types) if k not in rmCellTypes]
gene_dataset.filter_cell_types(newCellType)

if option == 'large':
    from scvi.dataset.dataset10X import Dataset10X
    dataset3 = Dataset10X('fresh_68k_pbmc_donor_a')
    dataset3.cell_types = np.asarray(['unlabelled'])
    gene_dataset = GeneExpressionDataset.concat_datasets(gene_dataset, dataset3)
    latent, batch_indices, labels, keys = run_model(model_type, gene_dataset, dataset1, dataset2, ngenes=5000)
elif option == 'small':
    latent, batch_indices, labels, keys = run_model(model_type, gene_dataset, dataset1, dataset2, ngenes=500)

if option=='large':
    latent = latent[batch_indices!=2]
    labels = labels[batch_indices!=2]
    keys = gene_dataset.cell_types[np.unique(labels)]
    _, labels = np.unique(labels,return_inverse=True)
    batch_indices = batch_indices[batch_indices!=2]
    batch_indices = batch_indices.reshape(len(batch_indices), 1)

eval_latent(batch_indices, labels, latent, keys, plotname+'.'+model_type)

