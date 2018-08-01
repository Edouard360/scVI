use_cuda = True
from rpy2.robjects.conversion import ri2py, py2ri
import rpy2.robjects.numpy2ri as numpy2ri
from scvi.harmonization.utils_chenling import get_matrix_from_dir
from scvi.harmonization.benchmark import assign_label
import numpy as np
from scipy.sparse import csr_matrix

from scvi.dataset.dataset10X import Dataset10X
from scvi.dataset.dataset import GeneExpressionDataset

from scvi.models.vae import VAE
from scvi.models.svaec import SVAEC

from scvi.inference.variational_inference import VariationalInference
from scvi.inference.variational_inference import SemiSupervisedVariationalInference
from scvi.metrics.clustering import get_latent
from scvi.metrics.classification import compute_accuracy

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

# if model_type == 'vae':
vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, n_labels=gene_dataset.n_labels,
          n_hidden=128, n_latent=10, n_layers=2, dispersion='gene')
infer_vae = VariationalInference(vae, gene_dataset, use_cuda=use_cuda)
infer_vae.train(n_epochs=250)
data_loader = infer_vae.data_loaders['sequential']
latent, batch_indices, labels = get_latent(vae, data_loader)
keys = gene_dataset.cell_types
batch_indices = np.concatenate(batch_indices)
keys = gene_dataset.cell_types
# elif model_type == 'svaec':
svaec = SVAEC(gene_dataset.nb_genes, gene_dataset.n_batches,
              gene_dataset.n_labels,use_labels_groups=False,
              n_latent=10,n_layers=2)
infer = SemiSupervisedVariationalInference(svaec, gene_dataset)
infer.train(n_epochs=50)
infer.accuracy('unlabelled')
data_loader = infer.data_loaders['unlabelled']
latent, batch_indices, labels = get_latent(infer.model, infer.data_loaders['unlabelled'])
keys = gene_dataset.cell_types
batch_indices = np.concatenate(batch_indices)
# elif model_type == 'Seurat':
SEURAT = SEURAT()
SEURAT.create_seurat(dataset1, 1)
SEURAT.create_seurat(dataset2, 2)
latent, batch_indices,labels,cell_types = SEURAT.get_cca()
numpy2ri.activate()
latent  = ri2py(latent)
batch_indices  = ri2py(batch_indices)
labels  = ri2py(labels)
keys,labels = np.unique(labels,return_inverse=True)
latent  = np.array(latent)
batch_indices  = np.array(batch_indices)
labels = np.array(labels)
# elif model_type == 'Combat':
COMBAT = COMBAT()
# corrected = COMBAT.combat_correct(gene_dataset)
latent = COMBAT.combat_pca(gene_dataset)
latent = latent.T
batch_indices = np.concatenate(gene_dataset.batch_indices)
labels = np.concatenate(gene_dataset.labels)
keys = gene_dataset.cell_types



sample = select_indices_evenly(2000,batch_indices)
batch_entropy = entropy_batch_mixing(latent[sample, :], batch_indices[sample])
print("Entropy batch mixing :", batch_entropy)

if model_type == 'svaec':
    print('svaec acc = ',infer.accuracy('unlabelled'))

labels = labels.astype('int')
sample = select_indices_evenly(100,labels)

# knn = knn_purity_avg(
#     latent[sample, :], labels[sample].astype('int'),
#     keys=keys, acc=True
# )
# print('average classification accuracy per cluster')
# for x in knn:
#     print(x)
#
# knn_acc = np.mean([x[1] for x in res])
# print("average KNN accuracy:", knn_acc)

sample = select_indices_evenly(1000,labels)
latent_s = latent[sample, :]
batch_s = batch_indices[sample]
label_s = labels[sample]
if latent_s.shape[1] != 2:
    latent_s = TSNE().fit_transform(latent_s)


plt.figure(figsize=(10, 10))
plt.scatter(latent_s[:, 0], latent_s[:, 1], c=batch_s, edgecolors='none')
plt.axis("off")
plt.tight_layout()
plt.savefig('../' + plotname + '.' + model_type + '.batch.png')

colors = sns.color_palette('tab20',len(unique(labels_s)))
fig, ax = plt.subplots(figsize=(10, 10))
for i, k in enumerate(keys):
    ax.scatter(latent_s[label_s == i, 0], latent_s[label_s == i, 1], c=colors[i], label=k, edgecolors='none')

ax.legend()
fig.tight_layout()
fig.savefig('../' + plotname + '.' + model_type + '.label.png', dpi=fig.dpi)

res = clustering_scores(np.asarray(latent)[sample,:],labels[sample],'knn',len(np.unique(labels[sample])))
for x in res:
    print(x,res[x])

