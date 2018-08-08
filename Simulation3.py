use_cuda = True
from scvi.harmonization.utils_chenling import eval_latent, run_model

import numpy as np
from scipy.sparse import csr_matrix

from scvi.dataset.dataset import GeneExpressionDataset
import sys

model_type = str(sys.argv[1])
plotname = 'simulation.EVF'

count = np.load('../sim_data/Sim_EVFbatch.UMI.npy')
count = count.T
meta = np.load('../sim_data/Sim_EVFbatch.meta.npy')

dataset1 = GeneExpressionDataset(
    *GeneExpressionDataset.get_attributes_from_matrix(
        csr_matrix(count[meta[:, 2] == 0, :]), labels=meta[meta[:, 2] == 0, 1]),
    gene_names=['gene' + str(i) for i in range(2000)], cell_types=['type' + str(i + 1) for i in range(5)])

dataset2 = GeneExpressionDataset(
    *GeneExpressionDataset.get_attributes_from_matrix(
        csr_matrix(count[meta[:, 2] == 1, :]), labels=meta[meta[:, 2] == 1, 1]),
    gene_names=['gene' + str(i) for i in range(2000)], cell_types=['type' + str(i + 1) for i in range(5)])

gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)

latent, batch_indices, labels,keys = run_model(model_type, gene_dataset, dataset1, dataset2)
eval_latent(batch_indices, labels, latent, keys, plotname+'.'+model_type)

#
# if model_type == 'vae':
#     vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, n_labels=gene_dataset.n_labels,
#               n_hidden=128, n_latent=10, n_layers=2, dispersion='gene')
#     infer = VariationalInference(vae, gene_dataset, use_cuda=use_cuda)
#     infer.train(n_epochs=250)
#     data_loader = infer.data_loaders['sequential']
#     latent, batch_indices, labels = get_latent(vae, data_loader)
#     keys = gene_dataset.cell_types
#     batch_indices = np.concatenate(batch_indices)
#     keys = gene_dataset.cell_types
# elif model_type == 'svaec':
#     svaec = SVAEC(gene_dataset.nb_genes, gene_dataset.n_batches,
#                   gene_dataset.n_labels, use_labels_groups=False,
#                   n_latent=10, n_layers=2)
#     infer = SemiSupervisedVariationalInference(svaec, gene_dataset)
#     infer.train(n_epochs=50)
#     infer.accuracy('unlabelled')
#     data_loader = infer.data_loaders['unlabelled']
#     latent, batch_indices, labels = get_latent(infer.model, infer.data_loaders['unlabelled'])
#     batch_indices = np.concatenate(batch_indices)
#     keys = gene_dataset.cell_types
# elif model_type == 'Seurat':
#     SEURAT = SEURAT()
#     seurat1 = SEURAT.create_seurat(dataset1, 1)
#     seurat2 = SEURAT.create_seurat(dataset2, 2)
#     latent, batch_indices, labels, keys = SEURAT.get_cca()
# elif model_type == 'Combat':
#     COMBAT = COMBAT()
#     latent = COMBAT.combat_pca(gene_dataset)
#     latent = latent.T
#     batch_indices = np.concatenate(gene_dataset.batch_indices)
#     labels = np.concatenate(gene_dataset.labels)
#     keys = gene_dataset.cell_types
# elif model_type == 'PCA':
#     pca = PCA(n_components=10)
#     pca.fit(gene_dataset.X.T.todense())
#     latent = pca.components_.T
#     batch_indices = np.concatenate(gene_dataset.batch_indices)
#     labels = np.concatenate(gene_dataset.labels)
#     keys = gene_dataset.cell_types
# elif model_type =='EVF':
#     pca = PCA(n_components=10)
#     latent = np.load('../sim_data/Sim_EVFbatch.evf.npy')
#     batch_indices = meta[:,2]
#     labels = meta[:,1]
#     pca.fit(latent.T)
#     latent = pca.components_.T
#     keys = gene_dataset.cell_types


