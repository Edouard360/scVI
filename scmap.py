import warnings

import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.rinterface import RRuntimeWarning
from sklearn.metrics import cohen_kappa_score

# TODO: at https://github.com/hemberg-lab/scRNA.seq.datasets/tree/master/bash
# execute segerstolpe.sh
# execute muraro.sh

# TODO: then execute the R scripts at https://github.com/hemberg-lab/scRNA.seq.datasets/blob/master/R
# Just beware of the path mappings

class SCMAP():
    def __init__(self):
        # self.n_clusters = n_clusters
        warnings.filterwarnings("ignore", category=RRuntimeWarning)
        rpy2.robjects.numpy2ri.activate()
        ro.r["library"]("scmap")
        ro.r["library"]("SingleCellExperiment")
        ro.r["library"]("matrixStats")

    def create_sce_object(self, matrix, gene_names, labels, name):
        n_samples, nb_genes = matrix.shape
        r_matrix = ro.r.matrix(matrix, nrow=nb_genes, ncol=n_samples)
        ro.r.assign("counts", r_matrix)
        ro.r.assign("gene_names", ro.StrVector(gene_names))
        ro.r("counts<-as.data.frame(counts, row.names=gene_names)")

        ro.r.assign("barcodes_cells", ro.StrVector(["cell_" + str(i) for i in range(n_samples)]))
        ro.r("colnames(counts)<-barcodes_cells")

        if labels:
            ro.r.assign("labels", ro.StrVector(labels))
            ro.r("barcodes_cells<-as.data.frame(labels, row.names=barcodes_cells, col.names=c('cell_type1'))")
            ro.r("colnames(barcodes_cells)<-c('cell_type1')")
            ro.r("%s <- SingleCellExperiment(assays=list(counts=as.matrix(counts)), colData=barcodes_cells)" % name)
        else:
            ro.r("%s <- SingleCellExperiment(assays=list(counts=as.matrix(counts)))" % name)
        ro.r("rowData(%s)$feature_symbol<-rownames(%s)" % (name, name))  # For any new custom dataset.
        ro.r("logcounts(%s) <- log2(counts(%s) + 1)" % (name, name))
        print("SCE object : %s created" % name)

    def scmap_cluster(self, reference, projection, threshold=0.5, n_features=500):
        ro.r("%s<-selectFeatures(%s,  n_features=%d)" % (reference, reference, n_features))
        scmap_features = ro.r("rowData(%s)$scmap_features" % reference)
        print("%i/%i features selected" % (np.sum(scmap_features), len(scmap_features)))

        ro.r("%s<-indexCluster(%s)" % (reference, reference))
        ro.r("result<-scmapCluster(%s, list(metadata(%s)$scmap_cluster_index), threshold=%.1f)"
             % (projection, reference, threshold))  # list(metadata(sce_reference)$scmap_cluster_index))")

        self.probs = ro.r("result$scmap_cluster_siml")
        self.labels_pred = ro.r("result$scmap_cluster_labs")  # 'unassigned' are included
        self.combined_labels_pred = ro.r("result$combined_labs")

        return self.labels_pred

    def get_labels(self, name):
        return ro.r("colData(%s)$cell_type1" % name)

    @staticmethod
    def load_rds_file(name, filename):
        ro.r("%s <- readRDS('%s')" % (name, filename))


scmap = SCMAP()

# Scenario 1

nb_genes = 1000
n_samples = 2000

matrix = np.random.randint(42, size=(n_samples, nb_genes))
labels = [str(int(i)) for i in np.random.randint(3, size=(n_samples))]
gene_names = ['gene_' + str(i) for i in range(nb_genes)]

n_new_samples=1000
new_matrix = np.random.randint(42, size=(n_new_samples, nb_genes))

scmap.create_sce_object(matrix, gene_names, labels, 'sce_reference')
scmap.create_sce_object(new_matrix, gene_names, None, 'sce_to_project')
scmap.scmap_cluster('sce_reference', 'sce_to_project')

# Scenario 2 - Reproducing the papers benchmarks (sanity check)

'''
scmap.load_rds_file("muraro", "../scmap/muraro/muraro.rds")
scmap.load_rds_file("segerstolpe", "../scmap/segerstolpe/segerstolpe.rds")
# scmap.load_rds_file("baron","../scmap/baron-human/baron-human.rds")
# scmap.load_rds_file("xin","../scmap/xin/xin.rds")
labels_pred = scmap.scmap_cluster("muraro", "segerstolpe")
labels = scmap.get_labels("segerstolpe")
labels_pred=labels_pred[labels != "not applicable"]
labels=labels[labels != "not applicable"]

uns_rate = np.mean(labels_pred == "unassigned")
kappa = cohen_kappa_score(labels, labels_pred)
'''
