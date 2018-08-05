from scvi.dataset import GeneExpressionDataset

use_cuda = True
import math
from numpy.random import uniform
from copy import deepcopy
from scvi.harmonization.benchmark import sample_by_batch
from scvi.harmonization.utils_chenling import get_matrix_from_dir
from scvi.harmonization.benchmark import assign_label
import numpy as np
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.models.vae import VAE
from scvi.inference.variational_inference import VariationalInference
from scvi.harmonization.benchmark import knn_purity_avg
from scvi.metrics.clustering import select_indices_evenly,entropy_batch_mixing,clustering_scores

import sys

low = float(sys.argv[1])
high = float(sys.argv[2])
option = str(sys.argv[3])
method = str(sys.argv[4])
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
cell_type = ["CD4+ T Helper2", "CD56+ NK", "CD14+ Monocyte", "CD19+ B", "CD8+ Cytotoxic T", "FCGR3A Monocyte", "na"]
dataset2 = assign_label(cellid, geneid, labels_map, count, cell_type, seurat)
gene_dataset: GeneExpressionDataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
gene_dataset.subsample_genes(5000)

rmCellTypes = {'na', 'dendritic'}
newCellType = [k for i, k in enumerate(gene_dataset.cell_types) if k not in rmCellTypes]
gene_dataset.filter_cell_types(newCellType)

if option == 'large':
    from scvi.dataset.dataset10X import Dataset10X
    dataset3 = Dataset10X('fresh_68k_pbmc_donor_a')

for prob_i in range(50):
    correlation = low - 0.01
    labels = np.concatenate(gene_dataset.labels)
    batch_id = np.concatenate(gene_dataset.batch_indices)
    cell_types = gene_dataset.cell_types
    cellid = np.arange(0, len(batch_id))
    while (correlation < low or correlation > high):
        count = []
        cells = []
        for batch in [0, 1]:
            prob = [uniform(0, 1) for rep in range(len(cell_types))]
            freq = [np.sum(labels[batch_id == batch] == i) for i in np.unique(labels)]
            nsamples = [math.floor(freq[i] * prob[i]) for i in range(len(cell_types))]
            nsamples = np.asarray(nsamples)
            sample = sample_by_batch(labels[batch_id == batch], nsamples)
            sample = cellid[batch_id == batch][sample]
            count.append(nsamples)
            cells.append(sample)
        correlation = (np.corrcoef(count[0], count[1])[0, 1])
    print("dataset 1 has %d cells" % (np.sum(count[0])))
    print("dataset 2 has %d cells" % (np.sum(count[1])))
    print("correlation between the cell-type composition of the subsampled dataset is %.3f" % correlation)
    sub_dataset = deepcopy(gene_dataset)
    sub_dataset.update_cells(np.concatenate(cells))
    if option == 'small':
        sub_dataset.subsample_genes(1000)
    elif option =='large':
        sub_dataset = GeneExpressionDataset.concat_datasets(sub_dataset, dataset3)
    if method == 'vae':
        vae = VAE(sub_dataset.nb_genes, n_batch=sub_dataset.n_batches, n_labels=sub_dataset.n_labels,
                  n_hidden=128, dispersion='gene',n_layers=2)
        infer = VariationalInference(vae, sub_dataset, use_cuda=use_cuda)
        infer.train(n_epochs=250)
        latent, batch_indices, labels = infer.get_latent('sequential')
    batch_indices = np.concatenate(batch_indices)
    sample = select_indices_evenly(2000,batch_indices)
    batch_entropy = entropy_batch_mixing(latent[sample, :], batch_indices[sample])
    print("Entropy batch mixing :", batch_entropy)
    keys = sub_dataset.cell_types
    sample = select_indices_evenly(1000, labels)
    res = knn_purity_avg(
        latent[sample, :], labels[sample].astype('int'),
        keys=keys, acc=True
    )
    for x in res:
        print(x)
    knn_acc = np.mean([x[1] for x in res])
    print("average KNN accuracy:", knn_acc)
    res = clustering_scores(np.asarray(latent)[sample, :], labels[sample], 'knn', len(np.unique(labels[sample])))
    for x in res:
        print(x, res[x])
    infer.show_t_sne(color_by="batches and labels",name='sequential',save_name='temp')
