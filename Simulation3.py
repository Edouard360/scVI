use_cuda = True

import numpy as np
from scipy.sparse import csr_matrix

from scvi.dataset.dataset import GeneExpressionDataset

from scvi.models.vae import VAE
from scvi.models.scanvi import SCANVI

from scvi.inference.variational_inference import VariationalInference
from scvi.inference.variational_inference import SemiSupervisedVariationalInference
from scvi.metrics.clustering import get_latent

from scvi.harmonization.benchmark import knn_purity_avg
from scvi.metrics.clustering import select_indices_evenly, entropy_batch_mixing, clustering_scores

import sys
from sklearn.decomposition import PCA

model_type = str(sys.argv[1])
plotname = 'simulation.EVF'

count = np.load('../sim_data/Sim_EVFbatch.UMI.npy')
count = count.T
meta = np.load('../sim_data/Sim_EVFbatch.meta.npy')

count_1 = count[meta[:, 2] == 0, :]
labels_1 = meta[meta[:, 2] == 0, 1]

count_2 = count[meta[:, 2] == 1, :]
labels_2 = meta[meta[:, 2] == 1, 1]

dataset1 = GeneExpressionDataset(
    *GeneExpressionDataset.get_attributes_from_matrix(
        csr_matrix(count_1), labels=labels_1),
    gene_names=['gene' + str(i) for i in range(2000)], cell_types=['type' + str(i + 1) for i in range(5)])

dataset2 = GeneExpressionDataset(
    *GeneExpressionDataset.get_attributes_from_matrix(
        csr_matrix(count_2), labels=labels_2),
    gene_names=['gene' + str(i) for i in range(2000)], cell_types=['type' + str(i + 1) for i in range(5)])

gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)

if model_type in ['vae', 'svaec', 'Seurat', 'Combat']:
    if model_type == 'vae':
        vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, n_labels=gene_dataset.n_labels,
                  n_hidden=128, n_latent=10, n_layers=2, dispersion='gene')
        infer = VariationalInference(vae, gene_dataset, use_cuda=use_cuda)
        infer.train(n_epochs=250)
        data_loader = infer.data_loaders['sequential']
        latent, batch_indices, labels = get_latent(vae, data_loader)
        keys = gene_dataset.cell_types
        batch_indices = np.concatenate(batch_indices)
        keys = gene_dataset.cell_types
    elif model_type == 'svaec':
        svaec = SCANVI(gene_dataset.nb_genes, gene_dataset.n_batches,
                       gene_dataset.n_labels, use_labels_groups=False,
                       n_latent=10, n_layers=2)
        infer = SemiSupervisedVariationalInference(svaec, gene_dataset)
        infer.train(n_epochs=50)
        infer.accuracy('unlabelled')
        data_loader = infer.data_loaders['unlabelled']
        latent, batch_indices, labels = get_latent(infer.model, infer.data_loaders['unlabelled'])
        batch_indices = np.concatenate(batch_indices)
        keys = gene_dataset.cell_types
    elif model_type == 'Seurat':
        from scvi.harmonization.clustering.seurat import SEURAT
        SEURAT = SEURAT()
        seurat1 = SEURAT.create_seurat(dataset1, 1)
        seurat2 = SEURAT.create_seurat(dataset2, 2)
        latent, batch_indices, labels, keys = SEURAT.get_cca()
    elif model_type == 'Combat':
        from scvi.harmonization.clustering.combat import COMBAT
        COMBAT = COMBAT()
        latent = COMBAT.combat_pca(gene_dataset)
        latent = latent.T
        batch_indices = np.concatenate(gene_dataset.batch_indices)
        labels = np.concatenate(gene_dataset.labels)
        keys = gene_dataset.cell_types
    elif model_type == 'PCA':
        pca = PCA(n_components=10)
        pca.fit(gene_dataset.X.T.todense())
        latent = pca.components_
        batch_indices = np.concatenate(gene_dataset.batch_indices)
        labels = np.concatenate(gene_dataset.labels)
        keys = gene_dataset.cell_types
    elif model_type =='EVF':
        latent = np.load('../sim_data/Sim_EVFbatch.evf.npy')
        batch_indices = meta[:,2]
        labels = meta[:,1]
        # pca.fit(latent)
        # latent = pca.components_
        keys = gene_dataset.cell_types

    batch_entropy = entropy_batch_mixing(latent.T, batch_indices)
    print("Entropy batch mixing :", batch_entropy)

    sample = select_indices_evenly(1000, labels)
    res = knn_purity_avg(
        latent[sample, :], labels[sample].astype('int'),
        keys=keys, acc=True
    )

    print('average classification accuracy per cluster')
    for x in res:
        print(x)

    knn_acc = np.mean([x[1] for x in res])
    print("average KNN accuracy:", knn_acc)

    res = clustering_scores(np.asarray(latent)[sample, :], labels[sample], 'knn', len(np.unique(labels[sample])))
    for x in res:
        print(x, res[x])
else:
    from scvi.harmonization import SCMAP
    scmap = SCMAP()
    with open('Simulation3-scmap.txt', 'w') as file:
        print("Starting scmap")
        for n_features in [100, 300, 500, 1000, 2000]:
            scmap.set_parameters(n_features=n_features)
            scmap.fit_scmap_cluster(count_1, labels_1.astype(np.int))
            line = "Score EVF1->EVF2:%.4f  [n_features = %d]\n" % (scmap.score(count_2, labels_2), n_features)
            file.write(line)

            scmap.fit_scmap_cluster(count_2, labels_2.astype(np.int))
            line = "Score EVF2->EVF1:%.4f  [n_features = %d]\n" % (scmap.score(count_1, labels_1), n_features)
            file.write(line)
