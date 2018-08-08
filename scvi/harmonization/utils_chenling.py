import numpy as np
from numpy import loadtxt
from scipy.io import mmread
import tables
import scipy.sparse as sparse

from scvi.dataset import SemiSupervisedDataLoaders
from scvi.harmonization.benchmark import knn_purity_avg
from scvi.inference import VariationalInference, SemiSupervisedVariationalInference
from scvi.metrics.clustering import select_indices_evenly, clustering_scores, entropy_batch_mixing
from scvi.models import VAE, SVAEC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import sys
use_cuda = True

def run_model(model_type, gene_dataset, dataset1, dataset2,filename='temp',nlayers=2,dispersion='gene',ngenes=5000):
    if model_type == 'vae':
        gene_dataset.subsample_genes(ngenes)
        vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, n_labels=gene_dataset.n_labels,
                  n_hidden=128, n_latent=10, n_layers=nlayers, dispersion=dispersion)
        infer = VariationalInference(vae, gene_dataset, use_cuda=use_cuda)
        infer.train(n_epochs=250)
        latent, batch_indices, labels = infer.get_latent('sequential')
        batch_indices = np.concatenate(batch_indices)
        keys = gene_dataset.cell_types
    elif model_type == 'svaec1':
        gene_dataset.subsample_genes(ngenes)
        svaec = SVAEC(gene_dataset.nb_genes, gene_dataset.n_batches,
                      gene_dataset.n_labels, use_labels_groups=False,
                      n_latent=10, n_layers=nlayers, dispersion=dispersion)
        infer = SemiSupervisedVariationalInference(svaec, gene_dataset)
        data_loaders = SemiSupervisedDataLoaders(gene_dataset)
        data_loaders['labelled'] = data_loaders(indices=(gene_dataset.batch_indices == 0).ravel())
        data_loaders['unlabelled'] = data_loaders(indices=(gene_dataset.batch_indices == 1).ravel())
        infer.data_loaders = data_loaders
        infer.classifier_inference.data_loaders['train'] = data_loaders['labelled']
        infer.train(n_epochs=250)
        print('svaec acc =', infer.accuracy('unlabelled'))
        latent, batch_indices, labels = infer.get_latent('all')
        keys = gene_dataset.cell_types
        batch_indices = np.concatenate(batch_indices)
    elif model_type == 'svaec2':
        gene_dataset.subsample_genes(ngenes)
        svaec = SVAEC(gene_dataset.nb_genes, gene_dataset.n_batches,
                      gene_dataset.n_labels, use_labels_groups=False,
                      n_latent=10, n_layers=nlayers, dispersion=dispersion)
        infer = SemiSupervisedVariationalInference(svaec, gene_dataset)
        data_loaders = SemiSupervisedDataLoaders(gene_dataset)
        data_loaders['labelled'] = data_loaders(indices=(gene_dataset.batch_indices == 1).ravel())
        data_loaders['unlabelled'] = data_loaders(indices=(gene_dataset.batch_indices == 0).ravel())
        infer.data_loaders = data_loaders
        infer.classifier_inference.data_loaders['train'] = data_loaders['labelled']
        infer.train(n_epochs=250)
        print('svaec acc =', infer.accuracy('unlabelled'))
        latent, batch_indices, labels = infer.get_latent('all')
        keys = gene_dataset.cell_types
        batch_indices = np.concatenate(batch_indices)
    elif model_type == 'svaec':
        gene_dataset.subsample_genes(ngenes)
        svaec = SVAEC(gene_dataset.nb_genes, gene_dataset.n_batches,
                      gene_dataset.n_labels, use_labels_groups=False,
                      n_latent=10, n_layers=nlayers, dispersion=dispersion)
        infer = SemiSupervisedVariationalInference(svaec, gene_dataset)
        infer.train(n_epochs=250)
        print('svaec acc =', infer.accuracy('unlabelled'))
        latent, batch_indices, labels = infer.get_latent('all')
        keys = gene_dataset.cell_types
        batch_indices = np.concatenate(batch_indices)
    elif model_type == 'Combat':
        from scvi.harmonization.clustering.combat import COMBAT
        combat = COMBAT()
        latent = combat.combat_pca(gene_dataset)
        latent = latent.T
        batch_indices = np.concatenate(gene_dataset.batch_indices)
        labels = np.concatenate(gene_dataset.labels)
        labels = labels.astype('int')
        keys = gene_dataset.cell_types
    elif model_type == 'Seurat':
        from scvi.harmonization.clustering.seurat import SEURAT
        seurat = SEURAT()
        seurat.create_seurat(dataset1, 1)
        seurat.create_seurat(dataset2, 2)
        latent, batch_indices, labels, keys = seurat.get_cca()
    elif model_type =='scmap':
        from scvi.harmonization.classification.scmap import SCMAP
        print("Starting scmap")
        scmap = SCMAP()
        scmap.set_parameters(n_features=500)
        count1 = np.asarray(gene_dataset.X[gene_dataset.batch_indices.ravel() == 0, :].todense())
        label1 = gene_dataset.labels[gene_dataset.batch_indices.ravel() == 0].ravel().astype('int')
        count2 = np.asarray(gene_dataset.X[gene_dataset.batch_indices.ravel() == 1, :].todense())
        label2 = gene_dataset.labels[gene_dataset.batch_indices.ravel() == 1].ravel().astype('int')
        scmap.fit_scmap_cluster(count1,label1)
        print("Score :", scmap.score(count2,label2))
        sys.exit()
    elif model_type =='writedata':
        from scipy.io import mmwrite
        count = gene_dataset.X
        genenames = gene_dataset.gene_names.astype('str')
        labels = gene_dataset.labels.ravel().astype('int')
        cell_types = gene_dataset.cell_types.astype('str')
        batchid = gene_dataset.batch_indices.ravel().astype('int')
        mmwrite(filename + '.X.mtx', count)
        np.savetxt(filename + '.celltypes.txt', cell_types, fmt='%s')
        np.save(filename + '.labels.npy', labels)
        np.save(filename + '.genenames.npy', genenames)
        np.save(filename + '.batch.npy', batchid)
        sys.exit()
    elif model_type =='readSeurat':
        latent = np.genfromtxt(filename + '.CCA.txt')
        labels = np.genfromtxt(filename + '.CCA.label.txt').astype('int')
        keys = np.genfromtxt(filename + '.celltypes.txt', dtype='str', delimiter=',').ravel().astype('str')
        batch_indices = np.genfromtxt(filename + '.CCA.batch.txt')
    return latent, batch_indices, labels, keys


def eval_latent(batch_indices, labels, latent, keys, plotname=None):
    sample = select_indices_evenly(5000, batch_indices)
    batch_entropy = entropy_batch_mixing(latent[sample, :], batch_indices[sample])
    print("Entropy batch mixing :", batch_entropy)
    sample = select_indices_evenly(10000, labels)
    res = knn_purity_avg(
        latent[sample, :], labels[sample],
        keys=keys, acc=True
    )
    print('average classification accuracy per cluster', np.mean([x[1] for x in res]))
    for x in res:
        print(x)
    res = clustering_scores(np.asarray(latent)[sample, :], labels[sample], 'knn', len(np.unique(labels[sample])))
    for x in res:
        print(x, res[x])
    if plotname is not None:
        colors = sns.color_palette('tab20')
        sample = select_indices_evenly(1000, labels)
        latent_s = latent[sample, :]
        batch_s = batch_indices[sample]
        label_s = labels[sample]
        if latent_s.shape[1] != 2:
            latent_s = TSNE().fit_transform(latent_s)
        plt.figure(figsize=(10, 10))
        plt.scatter(latent_s[:, 0], latent_s[:, 1], c=batch_s, edgecolors='none')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig('../'+plotname+'.batch.png')
        fig, ax = plt.subplots(figsize=(10, 10))
        for k in range(len(np.unique(label_s))):
            ax.scatter(latent_s[label_s == k, 0], latent_s[label_s == k, 1], c=colors[k%20], label=keys[k],
                       edgecolors='none')
        ax.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
        fig.tight_layout()
        plt.savefig('../'+plotname+'.labels.png')



def get_matrix_from_dir(dirname):
    geneid = loadtxt('../'+ dirname +'/genes.tsv',dtype='str',delimiter="\t")
    cellid = loadtxt('../'+ dirname + '/barcodes.tsv',dtype='str',delimiter="\t")
    count = mmread('../'+ dirname +'/matrix.mtx')
    return count, geneid, cellid



def get_matrix_from_h5(filename, genome):
    with tables.open_file(filename, 'r') as f:
        try:
            group = f.get_node(f.root, genome)
        except tables.NoSuchNodeError:
            print("That genome does not exist in this file.")
            return None
        gene_names = getattr(group, 'gene_names').read()
        barcodes = getattr(group, 'barcodes').read()
        data = getattr(group, 'data').read()
        indices = getattr(group, 'indices').read()
        indptr = getattr(group, 'indptr').read()
        shape = getattr(group, 'shape').read()
        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
        gene_names = gene_names.astype('<U18')
        barcodes = barcodes.astype('<U18')
        return gene_names, barcodes, matrix


def TryFindCells(dict, cellid, count):
    """

    :param dict: mapping from cell id to cluster id
    :param cellid: cell id
    :param count: count matrix in the same order as cell id (filter cells out if cell id has no cluster mapping)
    :return:
    """
    res = []
    new_count = []
    for i,key in enumerate(cellid):
        try:
            res.append(dict[key])
            new_count.append(count[i])
        except KeyError:
            continue
    new_count = sparse.vstack(new_count)
    return(new_count, np.asarray(res))

