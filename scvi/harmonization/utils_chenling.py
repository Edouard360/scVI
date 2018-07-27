import numpy as np
from numpy import loadtxt
from scipy.io import mmread
import tables
import scipy.sparse as sparse


def combine(list_matrices, list_genes, minthres=0):
    """
    :param list_matrices: a list of matrices with genes are rows
    :param list_genenames: a list of np.array of genenames
    :return: matrices where each row is one of those genes and set of shared expressed genes
    """
    list_matrices = [x.todense() for x in list_matrices]
    list_genes = [np.asarray(x) for x in list_genes]
    list_genes = [[str(y) for y in x ]for x in list_genes]
    allgenes = np.unique(np.concatenate(list_genes))
    for x in list_genes:
        allgenes = set(allgenes).intersection(x)
    allgenes=list(allgenes)
    combined = []
    for i in range(len(list_matrices)):
        data = dict(zip(list_genes[i], list_matrices[i]))
        temp = [data[x] for x in allgenes]
        temp = np.asarray(temp)
        temp = np.squeeze(temp)
        combined.append(temp)

    return allgenes,combined



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
        gene_ids = getattr(group, 'genes').read()
        gene_names = getattr(group, 'gene_names').read()
        barcodes = getattr(group, 'barcodes').read()
        data = getattr(group, 'data').read()
        indices = getattr(group, 'indices').read()
        indptr = getattr(group, 'indptr').read()
        shape = getattr(group, 'shape').read()
        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
        gene_names = gene_names.astype('<U18')
        barcodes = barcodes.astype('<U18')
        return gene_ids, gene_names, barcodes, matrix


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

