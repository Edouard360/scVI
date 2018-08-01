use_cuda = True
from rpy2.robjects.conversion import ri2py, py2ri
import rpy2.robjects.numpy2ri as numpy2ri

import numpy as np
from scipy.sparse import csr_matrix

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
from scvi.metrics.clustering import select_indices_evenly,entropy_batch_mixing



import sys
model_type = str(sys.argv[1])
plotname = 'simulation.UMI_nonUMI'

# countUMI = np.load('../sim_data/count1.npy')
# countnonUMI = np.load('../sim_data/count2.npy')
# labelUMI = np.load('../sim_data/label1.npy')
# labelnonUMI = np.load('../sim_data/label2.npy')


countUMI = np.load('../sim_data/count.UMI.npy')
countnonUMI = np.load('../sim_data/count.nonUMI.npy')
labelUMI = np.load('../sim_data/label.UMI.npy')
labelnonUMI = np.load('../sim_data/label.nonUMI.npy')

UMI = GeneExpressionDataset(
            *GeneExpressionDataset.get_attributes_from_matrix(
                csr_matrix(countUMI.T), labels=labelUMI),
            gene_names=['gene'+str(i) for i in range(2000)], cell_types=['type'+str(i+1) for i in range(5)])

nonUMI = GeneExpressionDataset(
            *GeneExpressionDataset.get_attributes_from_matrix(
                csr_matrix(countnonUMI.T), labels=labelnonUMI),
            gene_names=['gene'+str(i) for i in range(2000)], cell_types=['type'+str(i+1) for i in range(5)])

gene_dataset = GeneExpressionDataset.concat_datasets(UMI,nonUMI)

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
    infer.accuracy('unlabelled')
    data_loader = infer.data_loaders['unlabelled']
    latent, batch_indices, labels = get_latent(infer.model, infer.data_loaders['unlabelled'])
    keys = gene_dataset.cell_types
    batch_indices = np.concatenate(batch_indices)
    keys = gene_dataset.cell_types
elif model_type == 'Seurat':
    SEURAT = SEURAT()
    seurat1 = SEURAT.create_seurat(UMI, 0)
    seurat2 = SEURAT.create_seurat(nonUMI, 1)
    latent, batch_indices,labels = SEURAT.combine_seurat(seurat1, seurat2)
    numpy2ri.activate()
    latent  = ri2py(latent)
    batch_indices  = ri2py(batch_indices)
    labels  = ri2py(labels)
    keys,labels = np.unique(labels,return_inverse=True)
    latent  = np.array(latent)
    batch_indices  = np.array(batch_indices)
    labels = np.array(labels)
elif model_type == 'Combat':
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
    svaec_acc = compute_accuracy(svaec, data_loader, classifier=svaec.classifier)


sample = select_indices_evenly(1000,labels)
res = knn_purity_avg(
    latent[sample, :], labels[sample].astype('int'),
    keys=keys, acc=True
)

print('average classification accuracy per cluster')
for x in res:
    print(x)

knn_acc = np.mean([x[1] for x in res])
print("average KNN accuracy:", knn_acc)

sample = select_indices_evenly(1000,labels)
latent_s = latent[sample, :]
batch_s = batch_indices[sample]
label_s = labels[sample]
if latent_s.shape[1] != 2:
    latent_s = TSNE().fit_transform(latent_s)

colors = sns.color_palette('tab10',5)
fig, ax = plt.subplots(figsize=(10, 10))
for i, k in enumerate(keys):
    ax.scatter(latent_s[label_s == i, 0], latent_s[label_s == i, 1], c=colors[i], label=k, edgecolors='none')

ax.legend()
fig.tight_layout()
fig.savefig('../' + plotname + '.' + model_type + '.label.png', dpi=fig.dpi)

plt.figure(figsize=(10, 10))
plt.scatter(latent_s[:, 0], latent_s[:, 1], c=batch_s, edgecolors='none')
plt.axis("off")
plt.tight_layout()
plt.savefig('../' + plotname + '.' + model_type + '.batch.png')
