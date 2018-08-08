from scvi.dataset import SemiSupervisedDataLoaders
from scvi.models import SVAEC

use_cuda = True

from numpy.random import uniform
import numpy as np
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.dataset.BICCN import MacoskoDataset, RegevDataset
from copy import deepcopy
from scvi.models.vae import VAE
from scvi.inference import VariationalInference, SemiSupervisedVariationalInference
from scvi.harmonization.benchmark import sample_by_batch
from scvi.metrics.clustering import select_indices_evenly, entropy_batch_mixing, clustering_scores
from scvi.harmonization.benchmark import knn_purity_avg

import math
import sys

min = float(sys.argv[1])
max = float(sys.argv[2])
method = str(sys.argv[3])

dataset1 = MacoskoDataset()
dataset2 = RegevDataset()
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
gene_dataset.subsample_genes(5000)

batch_id = np.concatenate(gene_dataset.batch_indices)
cell_type = gene_dataset.cell_types
groups = ['Pvalb', 'L2/3', 'Sst', 'L5 PT', 'L5 IT Tcap', 'L5 IT Aldh1a7', 'L5 IT Foxp2', 'L5 NP',
          'L6 IT', 'L6 CT', 'L6 NP', 'L6b', 'Lamp5', 'Vip', 'Astro', 'OPC', 'VLMC', 'Oligo', 'Sncg', 'Endo',
          'SMC', 'MICRO']
cell_type = [x.upper() for x in cell_type]
groups = [x.upper() for x in groups]
labels = np.asarray([cell_type[x] for x in np.concatenate(gene_dataset.labels).astype('int')])
cell_type_bygroup = np.concatenate([[x for x in cell_type if x.startswith(y)] for y in groups])
new_labels_dict = dict(zip(cell_type_bygroup, np.arange(len(cell_type_bygroup))))
new_labels = np.asarray([new_labels_dict[x] for x in labels])
labels_groups = [[i for i, x in enumerate(groups) if y.startswith(x)][0] for y in cell_type_bygroup]
coarse_labels_dict = dict(zip(np.arange(len(labels_groups)), labels_groups))
coarse_labels = np.asarray([coarse_labels_dict[x] for x in new_labels])
gene_dataset.cell_types = np.asarray(groups)[np.unique(coarse_labels)]
gene_dataset.labels = np.unique(coarse_labels, return_inverse=True)[1].reshape(len(coarse_labels), 1)

for prob_i in range(5):
    correlation = min - 0.01
    while (correlation < min or correlation > max):
        cellid = np.arange(0, len(batch_id))
        cell_types = gene_dataset.cell_types
        labels = np.concatenate(gene_dataset.labels)
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
    if method =='vae':
        vae = VAE(sub_dataset.nb_genes, n_batch=sub_dataset.n_batches, n_labels=sub_dataset.n_labels,
                  n_hidden=128, dispersion='gene')
        infer = VariationalInference(vae, sub_dataset, use_cuda=use_cuda)
        infer.train(n_epochs=250)
        latent, batch_indices, labels = infer.get_latent('sequential')
    elif method == 'svaec':
        svaec = SVAEC(sub_dataset.nb_genes, sub_dataset.n_batches,
                      sub_dataset.n_labels)
        infer = SemiSupervisedVariationalInference(svaec, sub_dataset, verbose=True, classification_ratio=1,
                                                   n_epochs_classifier=1, lr_classification=5 * 10e-3, frequency=10)
        data_loaders = SemiSupervisedDataLoaders(sub_dataset)
        data_loaders['labelled'] = data_loaders(indices=(sub_dataset.batch_indices == i).ravel())
        data_loaders['unlabelled'] = data_loaders(indices=(sub_dataset.batch_indices == (1 - i)).ravel())
        infer.metrics_to_monitor = ['ll', 'accuracy', 'entropy_batch_mixing']
        infer.data_loaders = data_loaders
        infer.classifier_inference.data_loaders['train'] = data_loaders['labelled']
        infer.train(n_epochs=100)
        print('svaec acc =', infer.accuracy('unlabelled'))
        latent, batch_indices, labels = infer.get_latent('unlabelled')
        keys = gene_dataset.cell_types
        batch_indices = np.concatenate(batch_indices)
    keys = sub_dataset.cell_types
    batch_entropy = entropy_batch_mixing(latent, batch_indices)
    print("Entropy batch mixing :", batch_entropy)
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
