from scvi.benchmarks import SCMAP
import numpy as np
from scvi.dataset import *
from scvi.models import *
from scvi.inference import *


scmap = SCMAP()
# TODO 1 a) : scmap comparison: xin, segerstolpe, muraro, baron-human
dataset_xin = scmap.create_dataset("../scmap/xin/xin.rds")
cell_types_xin = list(filter(lambda p: ".contaminated" not in p, dataset_xin.cell_types))
dataset_xin.filter_cell_types(cell_types_xin)

dataset_segerstolpe = scmap.create_dataset("../scmap/segerstolpe/segerstolpe.rds")
cell_types_segerstolpe = list(filter(lambda p: p != "not applicable", dataset_segerstolpe.cell_types))
dataset_segerstolpe.filter_cell_types(cell_types_segerstolpe)

dataset_muraro = scmap.create_dataset("../scmap/muraro/muraro.rds")
dataset_baron = scmap.create_dataset("../scmap/baron-human/baron-human.rds")

concatenated = GeneExpressionDataset.concat_datasets(dataset_xin, dataset_segerstolpe, dataset_muraro, dataset_baron, on='gene_symbols')  #
concatenated.subsample_genes(subset_genes=(dataset_xin.X.max(axis=0) > 2500).ravel())
all_cell_types = list(filter(lambda p: p != "unknown", dataset_xin.cell_types))
concatenated.filter_cell_types(all_cell_types)

#concatenated.subsample_genes(2000)

# data_loaders_ = SemiSupervisedDataLoaders(dataset_xin, n_labelled_samples_per_class=10)
data_loaders = SemiSupervisedDataLoaders(concatenated, n_labelled_samples_per_class=10)
data_loaders['xin'] = data_loaders(indices=range(len(dataset_xin)))
data_loaders['labelled'] = data_loaders(indices=range(len(dataset_xin), len(concatenated)))
data_loaders['segerstolpe'] = data_loaders(indices=range(len(dataset_xin), len(dataset_segerstolpe) + len(dataset_xin)))
data_loaders['muraro'] = data_loaders(indices=range(len(dataset_segerstolpe) + len(dataset_xin), len(concatenated)))
data_loaders.to_monitor = ['xin', 'segerstolpe', 'muraro', 'sequential']
data_loaders.data_loaders_loop = ['all', 'labelled']

# len(concatenated)))#data_loaders_['labelled'].sampler.indices)#data_loaders_['labelled'].sampler.indices+len(dataset_xin))

# TODO 1 b) : scmap comparison: shekhar, macosko

dataset_shekhar = scmap.create_dataset("../scmap/shekhar/shekhar.rds")
dataset_macosko = scmap.create_dataset("../scmap/macosko/macosko.rds")

concatenated = GeneExpressionDataset.concat_datasets(dataset_shekhar,dataset_macosko, on='gene_symbols')

svaec = SVAEC(concatenated.nb_genes, concatenated.n_batches, concatenated.n_labels)

infer = JointSemiSupervisedVariationalInference(svaec, concatenated, frequency=5, verbose=True,
                                                classification_ratio=1000)
infer.data_loaders = data_loaders
infer.train(n_epochs=70)

(X_train, labels_train), = data_loaders.raw_data(data_loaders['segerstolpe'])
(X_test, labels_test), = data_loaders.raw_data(data_loaders['xin'])
scmap.create_sce_object(X_train, concatenated.gene_names, labels_train, 'd_train')
scmap.create_sce_object(X_test, concatenated.gene_names, labels_test, 'd_test')

# unique_train = np.unique()
#
# print("MAX Accuracy = ", np.mean([1 if l in unique_train else 0 for l in labels_test]))

labels_pred = scmap.scmap_cluster(
    'd_train', 'd_test', threshold=0, n_features=500
)
print("Accuracy result scmap_cluster:", scmap.accuracy_tuple(labels_test, labels_pred))

labels_pred = scmap.scmap_cell(
    'd_train', 'd_test', w=5, n_features=500, threshold=0
)
print("Accuracy result scmap_cell:", scmap.accuracy_tuple(labels_test, labels_pred))



#  TODO : THIS IS VERY FALSE !!!
# labels = scmap.get_labels('d_test')
# labels = labels.astype(np.int)
# labels = labels - 1  # labels get shifted from 0



''''''

''''''

# concatenated_xin = copy.deepcopy(concatenated)
# concatenated_segerstolpe = copy.deepcopy(concatenated)
# concatenated_xin.update_cells((concatenated.batch_indices == 1).ravel())
# concatenated_segerstolpe.update_cells((concatenated.batch_indices == 0).ravel())
# scmap.scmap_cluster_table(concatenated_xin, concatenated_segerstolpe, concatenated)


# xin	segerstolpe

# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()
#
# readRDS = robjects.r["readRDS"]
# df = readRDS("../scmap/xin/xin.rds")

# counts = counts.T
# #['.contaminated' in n for n in names]
# labels = ro.r("colData(xin)$cell_type1")
#
# df_xin = pandas2ri.ri2py(df)
#
# ro.r("segerstolpe<-readRDS('../scmap/segerstolpe/segerstolpe.rds')")
# df = readRDS("../scmap/segerstolpe/segerstolpe.rds")
# df_segerstolpe = pandas2ri.ri2py(df)
#


# Scenario 1 a) - Purified Populations
'''
from scvi.dataset import *
pure_dataset =PurePBMC()
pure_dataset.filter_cell_types(['CD4+/CD25 T Reg',
                                'CD4+/CD45RO+ Memory',
                                'CD4+/CD45RA+/CD25- Naive T',
                                'CD8+ Cytotoxic T'])
pure_dataset.subsample_genes(subset_genes=np.array(pure_dataset.X.sum(axis=0)>0)[0])
pure_dataset.subsample_genes(new_n_genes=3000)
pure_dataset.X = pure_dataset.X.A
scmap.scmap_cluster_self(pure_dataset)
# 0.60  ! That is promising
'''

# Scenario 1 b) - Loom Dataset
'''
import copy
from scvi.dataset import *

concatenated = LoomDataset('pbmc-pure-cite-B-CD4-CD14')
concatenated.subsample_genes(2000)

concatenated_donor = copy.deepcopy(concatenated)
concatenated_cite = copy.deepcopy(concatenated)
concatenated_donor.update_cells((concatenated.batch_indices == 1).ravel())
concatenated_cite.update_cells((concatenated.batch_indices == 0).ravel())

scmap.scmap_cluster_table(concatenated_cite, concatenated_donor, concatenated)

'''

# Scenario 2 - Synthetic Data

'''
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
'''

# Scenario 3 - Reproducing the papers benchmarks (sanity check)

'''
scmap.load_rds_file("muraro", "../scmap/muraro/muraro.rds")  # muraro is convenient (no preprocessing)
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




# concatenated_cite.subsample_cells(2000)
# concatenated_donor.subsample_cells(2000)

# [int(i) for i in concatenated_donor.labels.ravel()]
# from scvi.dataset import *
#
# data_loaders = SemiSupervisedDataLoaders(concatenated, n_labelled_samples_per_class=10)
# (train, test), = data_loaders.raw_data(data_loaders['labelled'])
#
# # scmap.create_sce_object(train, concatenated_cite.gene_names, test, 'cite')
# # scmap.create_sce_object(train, concatenated_cite.gene_names, test, 'donor')
# scmap.create_sce_object(train, concatenated_cite.gene_names, test, 'concatenated')
#
# scmap.create_sce_object(concatenated_donor.X, concatenated_donor.gene_names,
#                         [int(i) for i in concatenated_donor.labels.ravel()], 'donor')
# scmap.create_sce_object(concatenated_cite.X, concatenated_cite.gene_names, None,
#                         'cite')  # [int(i) for i in concatenated_cite.labels.ravel()]
#
# labels_pred = scmap.scmap_cluster('cite', 'donor', n_features=100, )
# labels_pred = scmap.scmap_cluster('concatenated', 'cite', n_features=100, threshold=0)
# print("ok")
# labels_pred[labels_pred == 'unassigned'] = -1
# labels_pred = labels_pred.astype(np.int)
#
# labels = concatenated_cite.labels.ravel().astype(np.int)
#
# np.mean(labels == labels_pred)
# 0.97
