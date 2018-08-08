from scvi.harmonization.utils_chenling import get_matrix_from_dir
use_cuda = True
import scvi.dataset.BICCN
from scvi.models.vae import VAE
from scvi.inference.variational_inference import VariationalInference
from scvi.harmonization.benchmark import assign_label
import numpy as np
count, geneid, cellid = get_matrix_from_dir('cite')
count = count.T.tocsr()
seurat = np.genfromtxt('../cite/cite.seurat.labels', dtype='str', delimiter=',')
cellid = scvi.dataset.BICCN.np.asarray([x.split('-')[0] for x in cellid])
labels_map = [0, 0, 1, 2, 3, 4, 5, 6]
cell_type = ["CD4+ T Helper2", "CD56+ NK", "CD14+ Monocyte", "CD19+ B", "CD8+ Cytotoxic T", "FCGR3A Monocyte", "na"]
dataset2 = assign_label(cellid, geneid, labels_map, count, cell_type, seurat)
dataset2.
np.save('cite.unfiltered.X.npy', dataset2.X)
np.save('cite.unfiltered.genenames.npy', geneid)

dataset2.subsample_genes(500)

vae = VAE(dataset2.nb_genes, n_batch=dataset2.n_batches, n_labels=dataset2.n_labels,
          n_hidden=128, n_latent=10, n_layers=2, dispersion='gene')
infer = VariationalInference(vae, dataset2, use_cuda=use_cuda)
infer.train(n_epochs=250)
data_loader = infer.data_loaders['sequential']
latent, batch_indices, labels = infer.get_latent('sequential')

np.save('cite.X.npy', dataset2.X)
np.save('cite.latent.npy', latent)
np.save('cite.genenames.npy', dataset2.gene_names)
labels = np.asarray([dataset2.cell_types[x] for x in labels])
np.save('cite.labels.npy', labels)

from scvi.dataset.cite_seq import CiteSeqDataset
gene_dataset = CiteSeqDataset('pbmc')

cellid1 = gene_dataset.expression.index
protein = gene_dataset.adt_expression_clr
protein = dict(zip(cellid1,protein))

prot = []
for x in seurat[1:,5]:
    prot.append(protein[x])

prot = np.vstack(prot)
np.save('cite.prot.npy', prot)
