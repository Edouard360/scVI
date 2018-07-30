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

'''
from scvi.models import *
from scvi.inference import *
from scvi.dataset import *
from scvi.dataset.mp_datasets import DonorPBMC, PurePBMC
from scvi.dataset import CiteSeqDataset

# 1 - Labelling CiteSeq # only for 'CD4', 'Bcells', 'CD14+'
citeseq = CiteSeqDataset('pbmc')
print(np.where(citeseq.protein_markers=='CD11c')[0][0]) # CD14+ # index = 8
print(np.where(citeseq.protein_markers=='CD4')[0][0]) # CD4 # index = 1
print(np.where(citeseq.protein_markers=='CD19')[0][0]) # CD19 # Bcells

plt.hist(citeseq.adt_expression_clr[:,8], bins=100)
plt.vlines(1,0, 250, color='r')
plt.title('CD11c - CD14+ Monocyte')
plt.ylim(0,450)
plt.savefig('CD11c.svg')

plt.hist(citeseq.adt_expression_clr[:,1], bins=100)
plt.title('CD4 - T cells')
plt.vlines(1.8,0, 250, color='r')
plt.ylim(0,450)
plt.savefig('CD4.svg')

citeseq.labels[(citeseq.adt_expression_clr[:,8]>=1)]=1 # CD11c -> CD14+
citeseq.labels[(citeseq.adt_expression_clr[:,1]>=1.8)]=2 # CD4
citeseq.labels[(citeseq.adt_expression_clr[:,9]>=2)]=3 # CD19 -> Bcells
citeseq.cell_types = np.array(["NA","CD14+","CD4","Bcells"])

# 2 - Merging the 3 datasets - The convoluted thing here is to manipulate indices btw cite, pure and donor

cite_seq_pbmc = CiteSeqDataset('pbmc')
cite_seq_pbmc.subsample_genes(3000)
donor_pbmc = DonorPBMC()
donor_pbmc.subsample_genes(3000)
pure_pbmc = PurePBMC()
pure_pbmc.subsample_genes(3000)
pure_pbmc.batch_indices[:]=0
pure_pbmc.n_batch=1

dataset = GeneExpressionDataset.concat_datasets(on='gene_symbols')
model = VAE(dataset.nb_genes, dataset.n_batches, dataset.n_labels)
# NB: for self labels prediction on the purified pop. restricted to tcells SVAEC(logreg_classifier=True)
# with AlternateSemiSupervisedInference is best.
infer = VariationalInference(model, dataset)
infer.train(200)

from scvi.dataset.mp_datasets import index_to_color
labels_color = [index_to_color[i][1] for i in range(pure_pbmc.n_labels)]
labels_name = [index_to_color[i][0] for i in range(pure_pbmc.n_labels)]

def _scatter(latent, labels=None, labels_color=None, labels_name=None, ax=None, prioritize=[]):
    if ax is None:
        ax=plt
    if labels is None:
        ax.scatter(latent[:, 0], latent[:, 1])
    else:
        n_labels = len(np.unique(labels))
        ordered_list = list(range(n_labels))
        for idx in prioritize:
            ordered_list.remove(idx)
        for idx in prioritize[::-1]:
            ordered_list.append(idx)
        for l in ordered_list:
            ax.scatter(latent[labels == l, 0], latent[labels == l, 1],
                       c=labels_color[l] if labels_color is not None else None,
                       label=labels_name[l] if labels_name is not None else None,
                       edgecolors='none', s=5)

def select_indices(latent, batch_indices, n_batch, n_samples):
    # returns all the indices, can be useful to get ADT counts
    latent_ = []
    batch_indices_ = []
    indices_ = []
    for i in range(n_batch):
        indices_i = np.where(batch_indices==i)[0]
        idx_permutation = np.random.permutation(len(indices_i))[:n_samples]
        new_indices_i = indices_i[idx_permutation]
        latent_+=[latent[new_indices_i]]
        batch_indices_+=[np.ones(n_samples)*i]
        indices_+=[new_indices_i]
    return np.concatenate(latent_), np.concatenate(batch_indices_), np.concatenate(indices_)

latent, batch_indices, labels = get_latent(vae, infer.data_loaders['sequential'])
print(latent.shape)
n_samples = 1000
latent_, batch_indices_, indices_ = select_indices(latent, batch_indices, n_batch=3, n_samples=1000)
print(latent_.shape)
latent__ = TSNE().fit_transform(latent_)
pure_indices_ = indices_[:1000]
labels = pure_pbmc.labels[pure_indices_].ravel()
latent_pure_ = latent__[:1000]

plt.figure(figsize=(12,8)) #
ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2,rowspan=2)
ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2,rowspan=2)
ax3 = plt.subplot2grid((3, 4), (2, 0))
ax4 = plt.subplot2grid((3, 4), (2, 1))
ax5 = plt.subplot2grid((3, 4), (2, 2))
ax6 = plt.subplot2grid((3, 4), (2, 3))

axes_cite_seq=[ax3,ax4,ax5,ax6]
n_samples=1000
indices_cite_seq = indices_[-1000:] - (len(pure_pbmc)+len(donor_pbmc))
expression_levels = cite_seq_pbmc.adt_expression_clr[indices_cite_seq]
latent_cite_seq__ = latent__[-1000:] # latent of the firsts 1000 are cite seq
mapping={
    'CD4':'CD4/CD8',
    'CD11c':'CD14/CD16',
    'CD19':'B',
    'CD16':'NK/CD16'
}

plt.suptitle('Harmonizing PBMC datasets')

colors = ["r",  # 1c86ee dodgerblue2,  # green 4
          "b",
         "gray"]  # cite seq
# pure in black, donor in red, cite seq in blue
for i, c, label in zip(reversed(range(3)),colors, ['Purified','Donor','citeSeq']):
    ax1.scatter(latent__[batch_indices_==i,0],latent__[batch_indices_==i,1],c=c,edgecolors='none', s=10, alpha=1, label = label)
ax1.set_xlim(-90,90)
ax1.set_ylim(-90,90)
ax1.axis('off')
ax1.set_title('donor (68 k) + purified(90k) + citeSeq (8k)')
ax1.legend()

_scatter(latent_pure_, labels, labels_color, labels_name, ax=ax2)
ax2.axis('off')
ax2.set_xlim(-90,90)
ax2.set_title('Purified populations')
ax2.set_ylim(-90,90)

for i, marker_name in enumerate(cite_seq_pbmc.protein_markers):
    to_do = ['CD4', 'CD11c',  'CD19', 'CD16']
    if marker_name in to_do:
        ax = axes_cite_seq[to_do.index(marker_name)]
        expression_level = expression_levels[:,i]
        ax.set_title(marker_name+' -> '+mapping[marker_name])
        ax.scatter(latent_cite_seq__[:,0],latent_cite_seq__[:,1],c=expression_level,cmap=plt.get_cmap('Reds'), s=3)
        ax.axis('off')
        ax.set_xlim(-90,90)
        ax.set_ylim(-90,90)
plt.savefig('new-harmonizing.svg')
'''
