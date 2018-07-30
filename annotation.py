from scvi.dataset import *
from scvi.dataset.data_loaders import SemiSupervisedDataLoaders
from scvi.harmonization import SCMAP, RF, SVC, COMBAT, SEURAT
from scvi.inference import *
from scvi.models import *
from scvi.metrics.clustering import clustering_scores
import numpy as np

# dataset=PurifiedPBMCDataset(filter_cell_types=range(6))
# dataset.subsample_genes(15000)

# dataset = SyntheticDataset()
#
# svaec = SVAEC(dataset.nb_genes, 0, dataset.n_labels, logreg_classifier=True)
# infer = AlternateSemiSupervisedVariationalInference(svaec, dataset, verbose=True, frequency=20, n_labelled_samples_per_class=100,n_epochs_classifier=1, lr_classification=1e-3)
# infer.train(n_epochs=500)
# infer.clustering_scores('unlabelled')

#scmap = SCMAP()

# TODO 1: ###################
# TODO 1: 1 Labelled Dataset

# Cortex / Synthetic / Purified / Any SCMAP dataset

# dataset = SyntheticDataset()
# dataset = CortexDataset()
#dataset=

# TODO 1-a: The VI models (might differ slightly across datasets)
n_epochs=400
class EXP():
    def run(self, dataset, name):
        # SVAEC - Start with SVAEC since it has the labelled data_loaders
        infer_svaec = SemiSupervisedVariationalInference(SVAEC(dataset.nb_genes, 0, dataset.n_labels), dataset,
                                                   n_labelled_samples_per_class=100)
        infer_svaec.train(n_epochs=n_epochs)
        infer_svaec.show_t_sne('sequential', color_by='labels', save_name='%s-svaec.svg'%name)
        infer_svaec.clustering_scores('unlabelled', prediction_algorithm='knn')
        infer_svaec.clustering_scores('unlabelled', prediction_algorithm='gmm')
        infer_svaec.clustering_scores('sequential', prediction_algorithm='knn')
        infer_svaec.clustering_scores('sequential', prediction_algorithm='gmm')
        infer_svaec.accuracy('unlabelled')
        infer_svaec.svc_latent_space()

        # VAE - update data_loaders - for fairness against SVAEC use all in train
        infer_vae = VariationalInference(VAE(dataset.nb_genes, 0), dataset)
        infer_vae.data_loaders.dict.update(infer_svaec.data_loaders.dict)
        infer_vae.data_loaders.loop = ['all']
        infer_vae.data_loaders.to_monitor = ['all']
        infer_vae.train(n_epochs=n_epochs)
        infer_vae.show_t_sne('sequential', color_by='labels', save_name='%s-vae.svg'%name)
        infer_vae.clustering_scores('sequential', 'knn')
        infer_vae.clustering_scores('sequential', prediction_algorithm='gmm')
        infer_vae.svc_latent_space()

        self.infer_svaec, self.infer_vae = infer_svaec, infer_vae

exp = EXP()
exp.run(SyntheticSimilar(), 'synthetic-similar')
exp.run(SyntheticDifferent(), 'synthetic-different')

# TODO 1-b: The benchmark accuracy algorithms : RF, SVC, SCMAP

# (X_train, y_train), (X_test, y_test) = DataLoaders.raw_data(
#     infer_svaec.data_loaders['labelled'], infer_svaec.data_loaders['unlabelled']
# )
# rf = RF()
# rf.fit(X_train, y_train)
# rf.score(X_test, y_test)
#
# svc = SVC()
# svc.fit(X_train, y_train)
# svc.score(X_test, y_test)

# TODO 1-c: The benchmark clustering algorithms : PCA, SIMLR, COMBAT

# TODO 2: ######################
# TODO 2: 2 Labelled Datasets
'''
import numpy as np
dataset_ = GeneExpressionDataset.load_pickle('../scmap/d4-all.pickle')
# ataset_ = GeneExpressionDataset.load_pickle('../scmap/shekhar-macosko-2.pickle')
# dataset_ = SyntheticDataset(batch_size=1000, n_batches=3)

index = {'xin': 0, 'segerstolpe': 1, 'muraro': 2, 'baron': 3}
# index={'shekhar':0, 'macosko':1}

# sources_target = [
#     ,
#
# import numpy as np
import copy
sources, target = ('segerstolpe', 'muraro')
if type(sources) is str:
    sources = [sources]
print("Sources : %s\nTarget : %s" % (' '.join(sources), target))
dataset = copy.deepcopy(dataset_)
dataset.update_cells((sum([dataset.batch_indices == index[s] for s in sources + [target]]).astype(np.bool).ravel()))
dataset.subsample_genes(1500)
dataset.squeeze()
print(dataset)

data_loaders = SemiSupervisedDataLoaders(dataset)
data_loaders['labelled'] = sum(data_loaders(indices=(dataset.batch_indices == index[source])) for source in sources)
data_loaders['unlabelled'] = data_loaders(indices=(dataset.batch_indices == index[target]).ravel())

(X_train, y_train), (X_test, y_test) = DataLoaders.raw_data(
    data_loaders['labelled'], data_loaders['unlabelled']
)
scmap.fit_scmap_cluster(X_train, y_train)
#scmap.predict_scmap_cluster(X_test, y_test)
print(scmap.score(X_test, y_test))

# dataset_xin = scmap.create_dataset("../scmap/xin/xin.rds")

# dataset_xin.export_loom('xin.loom')
# cell_types_xin = list(filter(lambda p: ".contaminated" not in p, dataset_xin.cell_types))
# dataset_xin.filter_cell_types(cell_types_xin)
#
# dataset_segerstolpe = scmap.create_dataset("../scmap/segerstolpe/segerstolpe.rds")
#
# dataset_segerstolpe.export_loom('segerstolpe.loom')
# # cell_types_segerstolpe = list(filter(lambda p: p != "not applicable", dataset_segerstolpe.cell_types))
# # dataset_segerstolpe.filter_cell_types(cell_types_segerstolpe)
#
# dataset_muraro = scmap.create_dataset("../scmap/muraro/muraro.rds")
# dataset_muraro.export_loom('muraro.loom')
# dataset_baron = scmap.create_dataset("../scmap/baron-human/baron-human.rds")
# dataset_baron.export_loom('baron.loom')


# dataset_shekhar = scmap.create_dataset("../scmap/shekhar/shekhar.rds")
# dataset_shekhar.export_loom('shekhar.loom')
#
# dataset_macosko = scmap.create_dataset("../scmap/macosko/macosko.rds")
# dataset_macosko.export_loom('macosko.loom')

# dataset = GeneExpressionDataset.concat_datasets(dataset_xin, dataset_segerstolpe, dataset_muraro, dataset_baron,
#                                                 on='gene_symbols')
# dataset.subsample_genes(subset_genes=(dataset.X.max(axis=0) <= 3000).ravel())
'''
'''
# dataset.update_cells(indices=.ravel())
# dataset.update_cells(subset_cells=((dataset.batch_indices == 2)+(dataset.batch_indices == 1)).ravel())
# print(dataset.X.shape)
# print(dataset.X)
# dataset.squeeze()
# data_loaders['xin'] = data_loaders(indices=(dataset.batch_indices == 0).ravel())
# data_loaders['segerstolpe'] = data_loaders(indices=(dataset.batch_indices == 1).ravel())
# data_loaders['muraro'] = data_loaders(indices=(dataset.batch_indices == 2).ravel())
# data_loaders['baron'] = data_loaders(indices=(dataset.batch_indices == 3).ravel())
# dataset.subsample_genes(1000)

from scvi.dataset import *
from scvi.inference import *
from scvi.models import *
import copy

# # TODO 0 a) : 2 datasets - they suck, we should filter them

# scmap = SCMAP()
# dataset_=SyntheticDataset(n_batches=3)
dataset_ = GeneExpressionDataset.load_pickle('../scmap/d4-all.pickle')
# ataset_ = GeneExpressionDataset.load_pickle('../scmap/shekhar-macosko-2.pickle')
# dataset_ = SyntheticDataset(batch_size=1000, n_batches=3)

index = {'xin': 0, 'segerstolpe': 1, 'muraro': 2, 'baron': 3}
# index={'shekhar':0, 'macosko':1}

sources_target = [
    ('segerstolpe', 'muraro'),
    # ('muraro', 'segerstolpe'),
    # (['baron-human', 'muraro'], 'segerstolpe'),
    # ('muraro', 'segerstolpe'),
    # ('segerstolpe', 'xin'),
    # ('xin', 'segerstolpe'),
    # (['xin','segerstolpe'], 'muraro')
    # ('shekhar', 'macosko')
]
import numpy as np

for sources, target in sources_target:
    if type(sources) is str:
        sources = [sources]
    print("Sources : %s\nTarget : %s" % (' '.join(sources), target))
    dataset = copy.deepcopy(dataset_)
    dataset.update_cells((sum([dataset.batch_indices == index[s] for s in sources + [target]]).astype(np.bool).ravel()))
    dataset.subsample_genes(1500)
    dataset.squeeze()
    print(dataset)

    data_loaders = SemiSupervisedDataLoaders(dataset)
    data_loaders['labelled'] = sum(data_loaders(indices=(dataset.batch_indices == index[source])) for source in sources)
    data_loaders['unlabelled'] = data_loaders(indices=(dataset.batch_indices == index[target]).ravel())
    data_loaders['all'] = data_loaders['labelled'] + data_loaders['unlabelled']
    for key, v in index.items():
        data_loaders[key] = data_loaders(indices=(dataset.batch_indices == v).ravel())

    # (X_train, labels_train), = data_loaders.raw_data(data_loaders['labelled'])
    # (X_test, labels_test), = data_loaders.raw_data(data_loaders['unlabelled'])
    ## X_train = X_train.A
    ## X_test = X_test.A
    # print(X_train.shape)
    # print(labels_train.shape)
    # print(X_test.shape)
    # print(labels_test.shape)

    # scmap.create_sce_object(X_train, dataset.gene_names, labels_train, 'd_train')
    # scmap.create_sce_object(X_test, dataset.gene_names, labels_test, 'd_test')
    # for n_features in [200,500,1000]:
    #     labels_pred = scmap.scmap_cluster(
    #         'd_train', 'd_test', threshold=0, n_features=n_features
    #     )
    #     print("Accuracy result scmap_cluster:", scmap.accuracy_tuple(labels_test, labels_pred))
    #
    # labels_pred = scmap.scmap_cell(
    #     'd_train', 'd_test', w=5, threshold=0, n_features=100
    # )
    # print("Accuracy result scmap_cell:", scmap.accuracy_tuple(labels_test, labels_pred))

    # # Optional: weighting the classification accuracy.
    data_loaders['labelled_weighted'] = data_loaders(indices=data_loaders['labelled'].sampler.indices, weighted=True)
    indices_unlabelled = data_loaders['unlabelled'].sampler.indices

    weights = np.array(data_loaders['labelled_weighted'].sampler.weights).astype(np.float32)
    weights[indices_unlabelled] = 1 / len(indices_unlabelled)
    data_loaders['all_weighted'] = data_loaders(weighted=weights)

    data_loaders.loop = ['all',
                         'labelled_weighted']  # ['all_weighted', 'labelled_weighted']#'] # maybe labelled_weighted ?

    svaec = SVAEC(dataset.nb_genes, dataset.n_batches, dataset.n_labels, n_layers=1, dropout_rate=0.1,
                  classifier_parameters={'n_layers': 2, 'dropout_rate': 0.1})
    # cls = Classifier(dataset.nb_genes, n_labels = dataset.n_labels)
    # infer = ClassifierInference(cls, dataset, verbose=True, frequency=50)
    infer = SemiSupervisedVariationalInference(svaec, dataset, frequency=100, verbose=True, classification_ratio=50,
                                               n_epochs_classifier=1, lr_classification=1e-3)
    # infer = VariationalInference(vae, dataset, frequency=10, verbose=True)#, classification_ratio=50,
    #                                            # n_epochs_classifier=1, lr_classification=1e-3)


    # infer = mmd_wrapper(infer, warm_up=0, scale=500)#gan_wrapper(infer, warm_up=200)

    infer.data_loaders = data_loaders

    infer.metrics_to_monitor = ['granular_accuracy', 'nn_latentspace', 'entropy_batch_mixing']
    infer.data_loaders.to_monitor = [target]

    # infer.classifier_inference.data_loaders['train'] = data_loaders['labelled_weighted']
    # infer.classifier_inference.data_loaders.loop = ['train']

    # infer.data_loaders['train'] = data_loaders['labelled']
    # infer.data_loaders['test'] = data_loaders['unlabelled']
    # infer.data_loaders.loop = ['train']

    infer.train(n_epochs=2000)
    print("Accuracy SVAEC : ", infer.accuracy(target))
    filename = '[%s]->%s' % ('-'.join(sources), target)

    # infer.classifier_inference.train(n_epochs=10, lr=1e-3)
    # print("Accuracy SVAEC : ", infer.accuracy(target))
    # infer.show_t_sne('all', color_by='batches and labels', save_name=filename+'.svg')
    # pickle.dump(dict(infer.history), open(filename+'.pickle', 'wb'))
    #
    # infer.classifier_inference.train(n_epochs=100, lr=1e-3)
    # print("Accuracy SVAEC : ", infer.accuracy(target))

infer.show_t_sne('sequential', color_by='batches and labels', n_samples=1000, save_name='mmd-penalty')
#
# dataset_shekhar = scmap.create_dataset("../scmap/shekhar/shekhar.rds")
# dataset_macosko = scmap.create_dataset("../scmap/macosko/macosko.rds")
#
# concatenated = GeneExpressionDataset.concat_datasets(dataset_shekhar,dataset_macosko, on='gene_symbols')
# #concatenated.subsample_genes(subset_genes=(concatenated.X.max(axis=0) <= 2500).ravel())
# all_cell_types = list(filter(lambda p: p not in ["unknown", "unclassified", "unclear"] , concatenated.cell_types))
# concatenated.filter_cell_types(all_cell_types)
# concatenated.X = csr_matrix(concatenated.X)
#
# concatenated.export_pickle('../scmap/shekhar-macosko-2.pickle')
# concatenated = GeneExpressionDataset.load_pickle('../scmap/shekhar-macosko-2.pickle')
# print(np.unique(concatenated.labels, return_counts=True))
# print(concatenated.cell_types)
#
# # TODO 1 a) : scmap comparison: xin, segerstolpe, muraro, baron-human

#
# concatenated = GeneExpressionDataset.concat_datasets(dataset_xin, dataset_segerstolpe, dataset_muraro, dataset_baron, on='gene_symbols')  #
# concatenated.subsample_genes(subset_genes=(dataset_xin.X.max(axis=0) > 2500).ravel())
# all_cell_types = list(filter(lambda p: p != "unknown", dataset_xin.cell_types))
# concatenated.filter_cell_types(all_cell_types)
#
# #concatenated.subsample_genes(2000)
#
# # data_loaders_ = SemiSupervisedDataLoaders(dataset_xin, n_labelled_samples_per_class=10)

#
# # len(concatenated)))#data_loaders_['labelled'].sampler.indices)#data_loaders_['labelled'].sampler.indices+len(dataset_xin))
#
# # TODO 1 b) : scmap comparison: shekhar, macosko
#
# dataset_shekhar = scmap.create_dataset("../scmap/shekhar/shekhar.rds")
# dataset_macosko = scmap.create_dataset("../scmap/macosko/macosko.rds")
#
# concatenated = GeneExpressionDataset.concat_datasets(dataset_shekhar,dataset_macosko, on='gene_symbols')
#

# infer.data_loaders = data_loaders
# infer.train(n_epochs=70)
#
# (X_train, labels_train), = data_loaders.raw_data(data_loaders['segerstolpe'])
# (X_test, labels_test), = data_loaders.raw_data(data_loaders['xin'])
# scmap.create_sce_object(X_train, concatenated.gene_names, labels_train, 'd_train')
# scmap.create_sce_object(X_test, concatenated.gene_names, labels_test, 'd_test')
#

# # unique_train = np.unique()
# #
# # print("MAX Accuracy = ", np.mean([1 if l in unique_train else 0 for l in labels_test]))
#
# labels_pred = scmap.scmap_cluster(
#     'd_train', 'd_test', threshold=0, n_features=500
# )
# print("Accuracy result scmap_cluster:", scmap.accuracy_tuple(labels_test, labels_pred))
#
# labels_pred = scmap.scmap_cell(
#     'd_train', 'd_test', w=5, n_features=500, threshold=0
# )
# print("Accuracy result scmap_cell:", scmap.accuracy_tuple(labels_test, labels_pred))



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

'''
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
