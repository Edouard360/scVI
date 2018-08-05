use_cuda = True
from rpy2.robjects.conversion import ri2py, py2ri
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as numpy2ri

from scvi.dataset.BICCN import *
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
plotname = 'Macosko_Regev'
dataset1 = MacoskoDataset()
dataset2 = RegevDataset()
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
gene_dataset.subsample_genes(5000)

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
    latent = np.genfromtxt('../macosko.CCA.txt')
    labels = np.genfromtxt('../macosko.CCA.label.txt').astype('int')
    keys = gene_dataset.cell_types
    batch_indices = np.genfromtxt('../macosko.CCA.batch.txt')
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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

colors = sns.color_palette('tab20')
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
plt.savefig('../easy2.Seurat.batch.png')

groups = ['Pvalb', 'L2/3', 'Sst', 'L5 PT', 'L5 IT Tcap', 'L5 IT Aldh1a7', 'L5 IT Foxp2', 'L5 NP',
                      'L6 IT', 'L6 CT', 'L6 NP', 'L6b', 'Lamp5', 'Vip', 'Astro', 'OPC', 'VLMC', 'Oligo', 'Sncg', 'Endo',
                      'SMC', 'MICRO']

fig, ax = plt.subplots(figsize=(20, 20))
for k in range(len(np.unique(label_s))):
    ax.scatter(latent_s[label_s == k, 0], latent_s[label_s == k, 1], c=colors[k],label=groups[k], edgecolors='none')

ax.legend()
fig.tight_layout()
fig.savefig('../easy2.Seurat.label.png', dpi=fig.dpi)

