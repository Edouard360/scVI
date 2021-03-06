use_cuda = True
from rpy2.robjects.conversion import ri2py, py2ri
import rpy2.robjects.numpy2ri as numpy2ri

import numpy as np
from scipy.sparse import csr_matrix

from scvi.dataset.dataset import GeneExpressionDataset

from scvi.models.vae import VAE
from scvi.models.scanvi import SCANVI

from scvi.inference import *

from scvi.metrics.clustering import get_latent
from scvi.metrics.classification import compute_accuracy

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from scvi.harmonization import SEURAT, COMBAT
from scvi.harmonization.benchmark import knn_purity_avg
from scvi.metrics.clustering import select_indices_evenly,entropy_batch_mixing,clustering_scores
from scvi.dataset.data_loaders import SemiSupervisedDataLoaders


import sys
model_type = str(sys.argv[1])
plotname = 'simulation.linear'

countUMI = np.load('../sim_data/count1.npy')
countnonUMI = np.load('../sim_data/count2.npy')
labelUMI = np.load('../sim_data/label1.npy')
labelnonUMI = np.load('../sim_data/label2.npy')

# model_type = str(sys.argv[1])
# plotname = 'simulation.uminonumi'
# countUMI = np.load('../sim_data/count.UMI.npy').T
# countnonUMI = np.load('../sim_data/count.nonUMI.npy').T
# labelUMI = np.load('../sim_data/label.UMI.npy').astype(np.int)
# labelnonUMI = np.load('../sim_data/label.nonUMI.npy').astype(np.int)

UMI = GeneExpressionDataset(
            *GeneExpressionDataset.get_attributes_from_matrix(
                csr_matrix(countUMI), labels=labelUMI),
            gene_names=['gene'+str(i) for i in range(2000)], cell_types=['type'+str(i+1) for i in range(5)])

nonUMI = GeneExpressionDataset(
            *GeneExpressionDataset.get_attributes_from_matrix(
                csr_matrix(countnonUMI), labels=labelnonUMI),
            gene_names=['gene'+str(i) for i in range(2000)], cell_types=['type'+str(i+1) for i in range(5)])


# countUMI = np.load('../sim_data/count.UMI.npy')
# countnonUMI = np.load('../sim_data/count.nonUMI.npy')
# labelUMI = np.load('../sim_data/label.UMI.npy')
# labelnonUMI = np.load('../sim_data/label.nonUMI.npy')
#
# UMI = GeneExpressionDataset(
#             *GeneExpressionDataset.get_attributes_from_matrix(
#                 csr_matrix(countUMI.T), labels=labelUMI),
#             gene_names=['gene'+str(i) for i in range(2000)], cell_types=['type'+str(i+1) for i in range(5)])
#
# nonUMI = GeneExpressionDataset(
#             *GeneExpressionDataset.get_attributes_from_matrix(
#                 csr_matrix(countnonUMI.T), labels=labelnonUMI),
#             gene_names=['gene'+str(i) for i in range(2000)], cell_types=['type'+str(i+1) for i in range(5)])


if model_type in ['vae', 'svaec', 'Seurat', 'Combat']:
    gene_dataset = GeneExpressionDataset.concat_datasets(UMI, nonUMI)
    if model_type == 'vae':
        vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, n_labels=gene_dataset.n_labels,
                  n_hidden=128, n_latent=10, n_layers=2, dispersion='gene')
        infer_vae = VariationalInference(vae, gene_dataset, use_cuda=use_cuda)
        infer_vae.train(n_epochs=250)
        data_loader = infer_vae.data_loaders['sequential']
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
        keys = gene_dataset.cell_types
        batch_indices = np.concatenate(batch_indices)
        keys = gene_dataset.cell_types

        for i in [0, 1]:
            svaec = SCANVI(gene_dataset.nb_genes, gene_dataset.n_batches,
                           gene_dataset.n_labels, n_layers=2)
            infer = SemiSupervisedVariationalInference(svaec, gene_dataset, verbose=True, classification_ratio=1,
                                                       n_epochs_classifier=1, lr_classification=5 * 10e-3, frequency=10)
            data_loaders = SemiSupervisedDataLoaders(gene_dataset)
            data_loaders['labelled'] = data_loaders(indices=(gene_dataset.batch_indices == i).ravel())
            data_loaders['unlabelled'] = data_loaders(indices=(gene_dataset.batch_indices == (1 - i)).ravel())
            infer.metrics_to_monitor = ['ll', 'accuracy', 'entropy_batch_mixing']
            infer.data_loaders = data_loaders
            infer.classifier_inference.data_loaders['train'] = data_loaders['labelled']
            infer.train(n_epochs=100)
            if i == 0:
                print("Score UMI->nonUMI:", infer.accuracy('unlabelled'))
            else:
                print("Score nonUMI->UMI:", infer.accuracy('unlabelled'))
    elif model_type == 'Seurat':
        from scvi.harmonization.clustering.seurat import SEURAT
        SEURAT = SEURAT()
        #The batch id number HAS to be 1 and 2
        seurat1 = SEURAT.create_seurat(UMI, 1)
        seurat2 = SEURAT.create_seurat(nonUMI, 2)
        latent, batch_indices,labels,keys = SEURAT.get_cca()
    elif model_type == 'Combat':
        from scvi.harmonization.clustering.combat import COMBAT
        COMBAT = COMBAT()
    # corrected = COMBAT.combat_correct(gene_dataset)
        latent = COMBAT.combat_pca(gene_dataset)
        latent = latent.T
        batch_indices = np.concatenate(gene_dataset.batch_indices)
        labels = np.concatenate(gene_dataset.labels)
        keys = gene_dataset.cell_types


    sample = select_indices_evenly(2000, batch_indices)
    batch_entropy = entropy_batch_mixing(latent[sample, :], batch_indices[sample])
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
elif model_type == 'scmap':
    from scvi.harmonization import SCMAP
    scmap = SCMAP()
    with open('Simulation1-scmap.txt', 'w') as file:
        print("Starting scmap")
        for n_features in [100, 300, 500, 1000, 2000]:
            scmap.set_parameters(n_features=n_features)
            scmap.fit_scmap_cluster(countUMI, labelUMI.astype(np.int))
            line = "Score 1->2:%.4f  [n_features = %d]\n" % (scmap.score(countnonUMI, labelnonUMI), n_features)
            file.write(line)

            scmap.fit_scmap_cluster(countnonUMI, labelnonUMI.astype(np.int))
            line = "Score 2->1:%.4f  [n_features = %d]\n" % (scmap.score(countUMI, labelUMI), n_features)
            file.write(line)
