use_cuda = True
from rpy2.robjects.conversion import ri2py, py2ri
import rpy2.robjects.numpy2ri as numpy2ri
from scvi.harmonization.utils_chenling import get_matrix_from_dir
from scvi.harmonization.benchmark import assign_label
import numpy as np

from scvi.dataset.dataset import GeneExpressionDataset

from scvi.models.vae import VAE
from scvi.models.svaec import SVAEC

from scvi.inference.variational_inference import VariationalInference
from scvi.inference.variational_inference import SemiSupervisedVariationalInference
from scvi.metrics.clustering import get_latent

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from scvi.harmonization.clustering.Seurat import SEURAT
from scvi.harmonization.clustering.Combat import COMBAT
from scvi.harmonization.benchmark import knn_purity_avg

from scvi.metrics.clustering import select_indices_evenly,entropy_batch_mixing,clustering_scores

import sys



model_type = str(sys.argv[1])
plotname = 'PBMC8_68k'
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
gene_dataset.subsample_genes(5000)

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
    svaec = SVAEC(gene_dataset.nb_genes, gene_dataset.n_batches,
                  gene_dataset.n_labels,use_labels_groups=False,
                  n_latent=10,n_layers=2)
    infer = SemiSupervisedVariationalInference(svaec, gene_dataset)
    infer.train(n_epochs=50)
    print('svaec acc =', infer.accuracy('unlabelled'))
    data_loader = infer.data_loaders['unlabelled']
    latent, batch_indices, labels = get_latent(infer.model, infer.data_loaders['unlabelled'])
    keys = gene_dataset.cell_types
    batch_indices = np.concatenate(batch_indices)
elif model_type == 'Seurat':
    SEURAT = SEURAT()
    seurat1 = SEURAT.create_seurat(dataset1, 1)
    seurat2 = SEURAT.create_seurat(dataset2, 2)
    latent, batch_indices,labels,keys = SEURAT.get_cca()
elif model_type == 'Combat':
    COMBAT = COMBAT()
    latent = COMBAT.combat_pca(gene_dataset)
    latent = latent.T
    batch_indices = np.concatenate(gene_dataset.batch_indices)
    labels = np.concatenate(gene_dataset.labels)
    keys = gene_dataset.cell_types



sample = select_indices_evenly(2000,batch_indices)
batch_entropy = entropy_batch_mixing(latent[sample, :], batch_indices[sample])
print("Entropy batch mixing :", batch_entropy)


sample = select_indices_evenly(1000,labels)
res = knn_purity_avg(
    latent[sample, :], labels[sample],
    keys=keys[np.unique(labels)], acc=True
)

print('average classification accuracy per cluster',np.mean([x[1] for x in res]))
for x in res:
    print(x)

res = clustering_scores(np.asarray(latent)[sample,:],labels[sample],'knn',len(np.unique(labels[sample])))
for x in res:
    print(x,res[x])

infer.show_t_sne(color_by="batches and labels")
