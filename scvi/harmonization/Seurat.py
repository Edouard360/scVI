import numpy as np
import rpy2.robjects as ro
import warnings
from rpy2.rinterface import RRuntimeWarning
import rpy2.robjects.numpy2ri as numpy2ri
from scipy.io import mmwrite

class SEURAT():
    def __init__(self):
        warnings.filterwarnings("ignore", category=RRuntimeWarning)
        numpy2ri.activate()
        r_source = ro.r['source']
        r_source("scvi/harmonization/R/Seurat.functions.R")
        ro.r["library"]("Matrix")

    def csr2r(self, matrix):
        # because rpy2 don't have sparse encoding try printing it to mtx and reading it in R
        # the object is named X
        mmwrite('temp.mtx',matrix)
        ro.r('X <- readMM("temp.mtx")')

    def create_seurat(self, dataset,batchname):
        genenames = dataset.gene_names
        genenames, uniq = np.unique(genenames,return_index=True)
        labels = [dataset.cell_types[int(i)] for i in np.concatenate(dataset.labels)]
        matrix = dataset.X[:,uniq]
        matrix = matrix.T.tocsr()
        self.csr2r(matrix)
        ro.r.assign("batchname", batchname)
        ro.r.assign("genenames", ro.StrVector(genenames))
        ro.r.assign("labels", ro.StrVector(labels))
        seurat = ro.r('SeuratPreproc(X,genenames,labels,batchname)')
        return seurat

    def combine_seurat(self, seurat1, seurat2):
        ro.r.assign("seurat1", seurat1)
        ro.r.assign("seurat2", seurat2)
        combined = ro.r('hvg_CCA(seurat1,seurat2)')
        return(combined)

    def get_cca(combined):
        ro.r.assign("combined", combined)
        command ='GetDimReduction(object=combined,' + \
        'reduction.type = "cca.aligned",' + \
        'slot = "cell.embeddings")'
        latent = ro.r(command)
        labels = ro.r('combined@meta.data$label')
        batch_indices = ro.r('combined@meta.data$batch')
        cell_types,labels = np.unique(labels,return_inverse=True)
        return latent,batch_indices,labels,cell_types

