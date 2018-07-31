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

    def csr2r(self, matrix):
        # because rpy2 don't have sparse encoding try printing it to mtx and reading it in R
        # the object is named X
        mmwrite('temp.mtx',matrix)
        ro.r('X <- readMM("temp.mtx")')

    def combat_correct(self, dataset):
        batch_indices = np.concatenate(dataset.batch_indices)
        ro.r.assign("batch", batch_indices)
        matrix = dataset.X.T.tocsr()
        self.csr2r(matrix)
        corrected = ro.r('ComBat(X,batch)')
        return corrected

    def combat_pca(self, dataset):
        corrected = self.combat_correct(dataset)
        pca = PCA(n_components=10)
        pca.fit(corrected)
        pc = pca.components_
        return pc

