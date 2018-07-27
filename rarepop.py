from scvi.harmonization.utils_chenling import get_matrix_from_dir
from scvi.harmonization.benchmark import assign_label, sample_by_batch, knn_purity_avg, sample_celltype
import numpy as np
from scvi.models.vae import VAE
from scvi.inference import VariationalInference
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.metrics.clustering import get_latent
from torch.utils.data import DataLoader
from scvi.metrics.clustering import entropy_batch_mixing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import mmwrite
import sys
from copy import deepcopy
use_cuda = True
from sklearn.datasets import load_svmlight_file
import pandas as pd

# subpop = "CD14+ Monocyte"
# prop = 0.1
# plotname = 'CD14_0.1'

subpop = str(sys.argv[1])
prop = float(sys.argv[2])
plotname = str(sys.argv[3])

print(subpop)

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

pbmc, pbmc_labels = load_svmlight_file("../68k_assignments.svmlight")
gene_symbols = pd.read_csv("../gene_symbols.csv").values.astype(np.str)[:, 1]
genenames = gene_symbols[1:-5]
temp = ["CD34+", "CD56+ NK", "CD4+/CD45RA+/CD25- Naive T", "CD4+/CD25 T Reg", "CD8+/CD45RA+ Naive Cytotoxic",
        "CD4+/CD45RO+ Memory", "CD8+ Cytotoxic T", "CD19+ B", "CD4+ T Helper2", "CD14+ Monocyte", "Dendritic"]
labels_map = [6, 1, 0, 0, 4, 0, 4, 3, 0, 2, 5]
cell_type = ["CD4+ T Helper2", "CD56+ NK", "CD14+ Monocyte", "CD19+ B", "CD8+ Cytotoxic T", "Dendritic", "CD34+"]
labels_new = deepcopy(pbmc_labels)
for i, j in enumerate(labels_map):
    labels_new[pbmc_labels == i] = j

dataset3 = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(pbmc.tocsr(), labels=labels_new),
                                 gene_names=genenames,
                                 cell_types=cell_type)

sub_dataset1 = sample_celltype(dataset1, subpop, prop)
print('total number of cells =' + str(
    [np.sum(sub_dataset1.labels == i) for i, k in enumerate(sub_dataset1.cell_types) if k == subpop][0]))
gene_dataset = GeneExpressionDataset.concat_datasets(sub_dataset1, dataset2, dataset3)
gene_dataset.subsample_genes(5000)

vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, n_labels=gene_dataset.n_labels,
          n_hidden=128, n_latent=10, n_layers=1, dispersion='gene')
infer_vae = VariationalInference(vae, gene_dataset, use_cuda=use_cuda)
infer_vae.fit(n_epochs=100)

np.save("../" + plotname + '.label.npy', gene_dataset.labels)
np.save("../" + plotname + '.batch.npy', gene_dataset.batch_indices)
mmwrite("../" + plotname + '.count.mtx', gene_dataset.X)

data_loader = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda, shuffle=False,
                         collate_fn=gene_dataset.collate_fn)
latent, batch_indices, labels = get_latent(infer_vae.model, data_loader)
keys = gene_dataset.cell_types
batch_indices = np.concatenate(batch_indices)

n_plotcells = 6000
pop1 = 0
pop2 = 1
nbatches = len(np.unique(batch_indices))
_, cell_count = np.unique(batch_indices, return_counts=True)

sample = sample_by_batch(batch_indices, int(n_plotcells / nbatches))
sample_2batch = sample[(batch_indices[sample] == pop1) + (batch_indices[sample] == pop2)]
batch_entropy = entropy_batch_mixing(latent[sample_2batch, :], batch_indices[sample_2batch])
print("Entropy batch mixing :", batch_entropy)

latent_s = latent[sample, :]
batch_s = batch_indices[sample]
label_s = labels[sample]
if latent_s.shape[1] != 2:
    latent_s = TSNE().fit_transform(latent_s)

plt.figure(figsize=(10, 10))
plt.scatter(latent_s[:, 0], latent_s[:, 1], c=batch_s, edgecolors='none')
plt.axis("off")
plt.tight_layout()
plt.savefig('../' + plotname + '.batch.png')

sample_2batch = np.random.permutation(sample_2batch)[:1000]

res1 = knn_purity_avg(
    latent[sample_2batch, :], labels[sample_2batch].astype('int'),
    gene_dataset.cell_types, acc=False
)
res2 = knn_purity_avg(
    latent[sample_2batch, :], labels[sample_2batch].astype('int'),
    gene_dataset.cell_types, acc=True
)
print('average KNN purity')
for x in res1:
    print(x)

print('average classification accuracy')
for x in res2:
    print(x)

# res1 = knn_purity(
#     latent[sample_2batch, :], labels[sample_2batch].astype('int'), batch_indices[sample_2batch], pop1, pop2,
#     gene_dataset.cell_types,acc=False
# )
# res2 = knn_purity(
#     latent[sample_2batch, :], labels[sample_2batch].astype('int'), batch_indices[sample_2batch], pop1, pop2,
#     gene_dataset.cell_types,acc=True
# )
# print('KNN purity of rare population in pop1')
# for x in res1:
#     print(x)
#
# print('classification accuracy of rare population in pop1')
# for x in res2:
#     print(x)

ordered_label = np.unique(label_s)
colors = sns.hls_palette(len(np.unique(label_s)))

fig, ax = plt.subplots(figsize=(10, 10))
for i, k in enumerate(ordered_label):
    ax.scatter(latent_s[label_s == k, 0], latent_s[label_s == k, 1], c=colors[i], label=keys[k], edgecolors='none')

ax.legend()
fig.tight_layout()
fig.savefig('../' + plotname + '.label.png', dpi=fig.dpi)
