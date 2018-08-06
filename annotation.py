# dataset=PurifiedPBMCDataset(filter_cell_types=range(6))
# dataset.subsample_genes(15000)

# dataset = SyntheticDataset()
#
# svaec = SVAEC(dataset.nb_genes, 0, dataset.n_labels, logreg_classifier=True)
# infer = AlternateSemiSupervisedVariationalInference(svaec, dataset, verbose=True, frequency=20, n_labelled_samples_per_class=100,n_epochs_classifier=1, lr_classification=1e-3)
# infer.train(n_epochs=500)
# infer.clustering_scores('unlabelled')

# scmap = SCMAP()

# TODO 1: ###################
# TODO 1: 1 Labelled Dataset

# Cortex / Synthetic / Purified / Any SCMAP dataset

# dataset = SyntheticDataset()
# dataset = CortexDataset()
# dataset=

# TODO 1-a: The VI models (might differ slightly across datasets)



'''
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
'''
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


from scvi.dataset import *

'''
'''
# # TODO 0 a) : 2 datasets - we should filter them
import pickle
from scvi.models import *
from scvi.inference import *

# scmap = SCMAP()

import os

filename_scmap = '4d-scmap-results.pickle'
if os.path.exists(filename_scmap):
    results_scmap = pickle.load(open(filename_scmap, 'rb'))
else:
    results_scmap = dict()

experiment_number = 11

filename_svaec = '4d-svaec-results-%i.pickle' % experiment_number
folder = 'figures_scmap_%i/' % experiment_number
if not os.path.exists(folder):
    os.makedirs(folder)

if os.path.exists(filename_svaec):
    results_svaec = pickle.load(open(filename_svaec, 'rb'))
else:
    results_svaec = dict()

dataset_ = GeneExpressionDataset.load_pickle('../scmap/d4-all.pickle')

all_cell_types = list(
    filter(lambda p: p not in ["unknown", "unclassified", "unclassified endocrine", "unclear"], dataset_.cell_types))
dataset_.filter_cell_types(all_cell_types)
index = {'xin': 0, 'segerstolpe': 1, 'muraro': 2, 'baron': 3}


# scmap = SCMAP()
names = ['xin', 'segerstolpe', 'muraro', 'baron']


def split_data(original_dataset, sources, target, nb_genes=1500):
    dataset = copy.deepcopy(original_dataset)
    dataset.update_cells(
        (sum([dataset.batch_indices == s for s in sources + (target,)]).astype(np.bool).ravel()))
    dataset.subsample_genes(nb_genes)
    dataset.squeeze()
    dataset.X = np.ascontiguousarray(dataset.X, dtype=np.float32)
    dataset.local_means = np.ascontiguousarray(dataset.local_means, dtype=np.float32)
    dataset.local_vars = np.ascontiguousarray(dataset.local_vars, dtype=np.float32)
    dataset.labels = np.ascontiguousarray(dataset.labels, dtype=np.float32)
    dataset.batch_indices = np.ascontiguousarray(dataset.batch_indices, dtype=np.float32)
    print("MAKING CONTIGUOUS")

    print(dataset)

    data_loaders = SemiSupervisedDataLoaders(dataset, batch_size=512) #batch_size=512 ?
    all_weights = np.ones(len(dataset))
    indices_source = sum((dataset.batch_indices == source).ravel() for source in sources).astype(np.bool)
    data_loaders['labelled'] = data_loaders(indices=indices_source)
    indices_target = (dataset.batch_indices == target).ravel().astype(np.bool)
    data_loaders['unlabelled'] = data_loaders(indices=indices_target)

    data_loaders['all'] = data_loaders['labelled'] + data_loaders['unlabelled']
    for key, v in index.items():
        data_loaders[key] = data_loaders(indices=(dataset.batch_indices == v).ravel())

    # all_weights[indices_source] = 1/np.sum(indices_source)
    # all_weights[indices_target] = 1/np.sum(indices_target)
    #
    # data_loaders['all_weighted'] = data_loaders(weights=all_weights)

    data_loaders.loop = ['all']#, 'labelled']  # _weighted
    # ['all_weighted', 'labelled_weighted']#'] # maybe labelled_weighted ?

    (X_train, labels_train), = data_loaders.raw_data(data_loaders['labelled'])
    (X_test, labels_test), = data_loaders.raw_data(data_loaders['unlabelled'])
    print("MAX Accuracy = ", np.mean([1 if l in np.unique(labels_train) else 0 for l in labels_test]))
    return dataset, data_loaders, (X_train, labels_train, X_test, labels_test)


# def svaec_experiment(dataset, data_loaders, results, title, sources, target):



# If you want to run a single experiment
# sources=(1,)
# target=0
# #


#
# infer.data_loaders.to_monitor = [names[target]]
# infer.metrics_to_monitor = ['accuracy','nn_latentspace', 'entropy_batch_mixing'] #
# infer.data_loaders = data_loaders
# infer.classifier_inference.data_loaders['train'] = infer.data_loaders['labelled']

# cls = Classifier(10, dataset.n_labels)
# infer_cls = ClassifierInference(cls,dataset,sampling_model=vae)



# infer = adversarial_wrapper(infer, warm_up=100)

# infer.train(400, lr=1e-3)
# infer.accuracy('labelled', verbose=True)
# print("further training")
# infer.classifier_inference.train(50)
# infer.accuracy('labelled',verbose=True)

# cls = Classifier(n_input=dataset.nb_genes, n_labels=dataset.n_labels)
# infer_cls = ClassifierInference(cls, dataset, verbose=True, frequency=10)
# infer_cls.metrics_to_monitor = ['accuracy']
# infer_cls.data_loaders['train'] = data_loaders['labelled']
# infer_cls.data_loaders['test'] = data_loaders['unlabelled']
# infer_cls.train(50, 1e-2)
# results[sources][target] = {'acc': infer.accuracy('unlabelled'),
#                             'bench_acc': infer.benchmark_accuracy('unlabelled'),
#                             'latent_space_acc': infer.nn_latentspace('unlabelled'),
#                             'entropy_batch_mixing': infer.entropy_batch_mixing('sequential')}
#
# infer.classifier_inference.train(20, lr=1e-2)
# results[sources][target]['acc_continued'] = infer.accuracy('unlabelled')
#
# infer.classifier_inference.train(20, lr=1e-2)
# results[sources][target]['acc_continued_2'] = infer.accuracy('unlabelled')
# print("TSNE !")
# infer.show_t_sne('sequential', color_by='batches and labels', n_samples=1000,
#                  save_name='figures_scmap_%i/%s.svg' % (experiment_number, title))


import copy
# If you want to run all other experiments
import itertools

# for grain in [500]:  # [100,500]:
#     scmap.set_parameters(n_features=grain)
#     results_scmap[grain] = result = dict()
nb_genes=600
import numpy as np

for i in range(3, 0,-1):
    for sources in list(itertools.combinations(range(0, 4), i)):
        targets = list(range(4))
        for source in sources:
            targets.remove(source)
        if sources not in results_svaec:
            results_svaec[sources] = dict()
        # if sources not in results_scmap[grain]:
        #     results_scmap[grain][sources] = dict()
        for target in targets:
            print("Sources : %s\nTarget : %s" % (' '.join([names[s] for s in sources]), names[target]))

            # svaec_experiment(dataset, data_loaders, results_svaec, title, sources, target)
            title = "%s -> %s" % (' '.join([names[s] for s in sources]), names[target])
            dataset, data_loaders, (X_train, labels_train, X_test, labels_test) = split_data(dataset_, sources,
                                                                                             target, nb_genes=nb_genes)  # , nb_genes=1500)
            # svaec_experiment(dataset, data_loaders, results_svaec, title, sources, target)

            svaec = SCANVI(dataset.nb_genes, dataset.n_batches, dataset.n_labels,
                           n_layers=2, dropout_rate=0.1, decoder_scvi_parameters={'n_layers': 2},
                           classifier_parameters={'n_layers': 2, 'n_hidden': 256})
                          # logreg_classifier=True
                          #decoder_scvi_parameters={'n_layers': 2}, # 2-3
                          #classifier_parameters={'n_layers': 3, 'n_hidden': 512})  # 3-1024 # 2 256
            # a priori not useful

            #infer = VariationalInference(svaec, dataset)

            # infer = SemiSupervisedVariationalInference(svaec, dataset, frequency=10, verbose=True,
            #                                            classification_ratio=1,
            #                                            n_epochs_classifier=1, lr_classification=5*1e-3)

            infer = AlternateSemiSupervisedVariationalInference(svaec, dataset, lr_classification=1e-2, n_epochs_classifier=1,
                                                                frequency=100, verbose=True)
            infer.data_loaders = data_loaders
            infer.classifier_inference.data_loaders['train'] = data_loaders['labelled']
            infer.metrics_to_monitor=['accuracy','ll','nn_latentspace']

            infer.train(800)
            results_svaec[sources][target] = {'acc': infer.accuracy('unlabelled'),
                                        'bench_acc': infer.benchmark_accuracy('unlabelled'),
                                        'latent_space_acc': infer.nn_latentspace('unlabelled'),
                                        'entropy_batch_mixing': infer.entropy_batch_mixing('sequential')}

            infer.classifier_inference.train(10)
            results_svaec[sources][target]['after_acc']=infer.accuracy('unlabelled')

            infer.classifier_inference.train(50)
            results_svaec[sources][target]['really_after_acc']=infer.accuracy('unlabelled')

            infer.show_t_sne('sequential',color_by='batches and labels', save_name=title+'.svg')

            #
            # infer = SemiSupervisedVariationalInference(svaec, dataset, frequency=10, verbose=True, classification_ratio=10,
            #                                                     n_epochs_classifier=1, lr_classification=1e-3)  # 1e-3,kl=0,

            pickle.dump(results_svaec, open(filename_svaec, 'wb'))

            # scmap.fit_scmap_cluster(X_train, labels_train)
            # score = scmap.score(X_test, labels_test)
            # print("Score: ", score)
            # results_scmap[grain][sources][target] = score
            #
            # pickle.dump(results_scmap, open(filename_scmap, 'wb'))
''''''

# TODO 2 - display results
# for sources in list(filter(lambda p: len(p)==3, results_svaec.keys())):
#     targets = list(range(4))
#     for source in sources:
#         targets.remove(source)
#     target = targets[0]
#     title = "%s -> %s" % (' '.join([names[s] for s in sources]), names[target])
#     print("Sources : %s\nTarget : %s" % (' '.join([names[s] for s in sources]), names[target]))
#     print(results_svaec[sources][target])


import numpy as np
import copy

# for sources, target in sources_target:
#     if type(sources) is str:
#         sources = [sources]


# X_train = X_train.A
# X_test = X_test.A

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
# data_loaders['labelled_weighted'] = data_loaders(indices=data_loaders['labelled'].sampler.indices, weighted=True)
# indices_unlabelled = data_loaders['unlabelled'].sampler.indices

# weights = np.array(data_loaders['labelled_weighted'].sampler.weights).astype(np.float32)
# weights[indices_unlabelled] = 1 / len(indices_unlabelled)
# data_loaders['all_weighted'] = data_loaders(weighted=weights)

# data_loaders.loop = ['all','labelled']  # ['all_weighted', 'labelled_weighted']#'] # maybe labelled_weighted ?
#
# svaec = SVAEC(dataset.nb_genes, dataset.n_batches, dataset.n_labels, n_layers=2, dropout_rate=0.1,
#               classifier_parameters={'n_layers': 2, 'dropout_rate': 0.1})
# # # cls = Classifier(dataset.nb_genes, n_labels = dataset.n_labels)
# # # infer = ClassifierInference(cls, dataset, verbose=True, frequency=50)
# infer = SemiSupervisedVariationalInference(svaec, dataset, frequency=100, verbose=True, classification_ratio=50,
#                                            n_epochs_classifier=800, lr_classification=1e-3)
# infer.benchmark_accuracy()
# # infer = VariationalInference(vae, dataset, frequency=10, verbose=True)#, classification_ratio=50,
# #                                            # n_epochs_classifier=1, lr_classification=1e-3)
#
#
# # infer = mmd_wrapper(infer, warm_up=0, scale=500)#gan_wrapper(infer, warm_up=200)
#
# infer.data_loaders = data_loaders
#
# infer.metrics_to_monitor = ['granular_accuracy', 'nn_latentspace', 'entropy_batch_mixing']
# infer.data_loaders.to_monitor = [target]
#
# # infer.classifier_inference.data_loaders['train'] = data_loaders['labelled_weighted']
# # infer.classifier_inference.data_loaders.loop = ['train']
#
# # infer.data_loaders['train'] = data_loaders['labelled']
# # infer.data_loaders['test'] = data_loaders['unlabelled']
# # infer.data_loaders.loop = ['train']
#
# infer.train(n_epochs=2000)
# print("Accuracy SVAEC : ", infer.accuracy(target))
# filename = '[%s]->%s' % ('-'.join(sources), target)

# infer.classifier_inference.train(n_epochs=10, lr=1e-3)
# print("Accuracy SVAEC : ", infer.accuracy(target))
# infer.show_t_sne('all', color_by='batches and labels', save_name=filename+'.svg')
# pickle.dump(dict(infer.history), open(filename+'.pickle', 'wb'))
#
# infer.classifier_inference.train(n_epochs=100, lr=1e-3)
# print("Accuracy SVAEC : ", infer.accuracy(target))

# infer.show_t_sne('sequential', color_by='batches and labels', n_samples=1000, save_name='mmd-penalty')
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
# from scvi.dataset import *

# pure_dataset =PurePBMC()
# pure_dataset.filter_cell_types(['CD4+/CD25 T Reg',
#                                 'CD4+/CD45RO+ Memory',
#                                 'CD4+/CD45RA+/CD25- Naive T',
#                                 'CD8+ Cytotoxic T'])
# pure_dataset.subsample_genes(subset_genes=np.array(pure_dataset.X.sum(axis=0)>0)[0])
# pure_dataset.subsample_genes(new_n_genes=3000)
# pure_dataset.X = pure_dataset.X.A
# scmap.scmap_cluster_self(pure_dataset)
# 0.60  ! That is promising

'''
# Scenario 1 b) - Loom Dataset
'''

# import copy
# from scvi.dataset import *
#
# concatenated = LoomDataset('pbmc-pure-cite-B-CD4-CD14')
# concatenated.subsample_genes(2000)
#
# concatenated_donor = copy.deepcopy(concatenated)
# concatenated_cite = copy.deepcopy(concatenated)
# concatenated_donor.update_cells((concatenated.batch_indices == 1).ravel())
# concatenated_cite.update_cells((concatenated.batch_indices == 0).ravel())
#
# scmap.scmap_cluster_table(concatenated_cite, concatenated_donor, concatenated)

'''

# Scenario 2 - Synthetic Data

'''
# nb_genes = 1000
# n_samples = 2000
#
# matrix = np.random.randint(42, size=(n_samples, nb_genes))
# labels = [str(int(i)) for i in np.random.randint(3, size=(n_samples))]
# gene_names = ['gene_' + str(i) for i in range(nb_genes)]
#
# n_new_samples=1000
# new_matrix = np.random.randint(42, size=(n_new_samples, nb_genes))
#
# scmap.create_sce_object(matrix, gene_names, labels, 'sce_reference')
# scmap.create_sce_object(new_matrix, gene_names, None, 'sce_to_project')
# scmap.scmap_cluster('sce_reference', 'sce_to_project')
'''

# Scenario 3 - Reproducing the papers benchmarks (sanity check)

'''
# scmap.load_rds_file("muraro", "../scmap/muraro/muraro.rds")  # muraro is convenient (no preprocessing)
# scmap.load_rds_file("segerstolpe", "../scmap/segerstolpe/segerstolpe.rds")
# # scmap.load_rds_file("baron","../scmap/baron-human/baron-human.rds")
# # scmap.load_rds_file("xin","../scmap/xin/xin.rds")
# labels_pred = scmap.scmap_cluster("muraro", "segerstolpe")
# labels = scmap.get_labels("segerstolpe")
# labels_pred=labels_pred[labels != "not applicable"]
# labels=labels[labels != "not applicable"]
#
# uns_rate = np.mean(labels_pred == "unassigned")
# kappa = cohen_kappa_score(labels, labels_pred)
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
#
'''
