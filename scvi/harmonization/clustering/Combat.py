import numpy as np
import rpy2.robjects as ro

import warnings
from rpy2.rinterface import RRuntimeWarning
import rpy2.robjects.numpy2ri as numpy2ri
from scipy.io import mmwrite
from sklearn.decomposition import PCA


class COMBAT():
    def __init__(self):
        warnings.filterwarnings("ignore", category=RRuntimeWarning)
        numpy2ri.activate()
        ro.r["library"]("gmodels")
        ro.r["library"]("sva")

    def combat_correct(self, dataset):
        batch_indices = np.concatenate(dataset.batch_indices)
        ro.r.assign("batch", ro.IntVector(batch_indices))
        X = np.asarray(dataset.X.T.todense())
        nb_genes, n_samples = X.shape
        X = ro.r.matrix(X, nrow=nb_genes, ncol=n_samples)
        ro.r.assign('X', X)
        corrected = ro.r('ComBat(X,batch)')
        return corrected

    def combat_pca(self, dataset):
        corrected = self.combat_correct(dataset)
        pca = PCA(n_components=10)
        pca.fit(corrected)
        pc = pca.components_
        return pc

