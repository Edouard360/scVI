import itertools
import os
import pickle
import copy
import numpy as np

from scvi.dataset import *
from scvi.dataset.scmap_datasets import XinDataset, SegerstolpeDataset, MuraroDataset, BaronDataset
from scvi.harmonization import SCMAP
# from scvi.inference import *
# from scvi.models import *
from scvi.dataset.synthetic import SyntheticUMI, SyntheticNONUMI

scmap = SCMAP()
synthetic_umi = SyntheticUMI()
synthetic_nonumi = SyntheticNONUMI()

for n_features in [100,300,1000]:
    scmap.set_parameters(n_features=n_features)
    scmap.fit_scmap_cluster(synthetic_umi.X, synthetic_nonumi.labels.astype(np.int).ravel())
    print(scmap.score(synthetic_nonumi.X, synthetic_nonumi.labels.astype(np.int).ravel()))

    scmap.set_parameters(n_features=n_features)
    scmap.fit_scmap_cluster(synthetic_nonumi.X, synthetic_nonumi.labels.astype(np.int).ravel())
    print(scmap.score(synthetic_umi.X, synthetic_nonumi.labels.astype(np.int).ravel()))




# results_svaec = dict()
#
# index = {'xin': 0, 'segerstolpe': 1, 'muraro': 2, 'baron': 3}
#
#
# def split_data(original_dataset, sources, target, nb_genes=1500, batch_size=128):
#     dataset = copy.deepcopy(original_dataset)
#     # dataset.update_cells(
#     #     (sum([dataset.batch_indices == s for s in sources + (target,)]).astype(np.bool).ravel()))
#     dataset.subsample_genes(nb_genes)
#     # dataset.squeeze()
#     #dataset = original_dataset
#     dataset.X = np.ascontiguousarray(dataset.X, dtype=np.float32)
#     dataset.local_means = np.ascontiguousarray(dataset.local_means, dtype=np.float32)
#     dataset.local_vars = np.ascontiguousarray(dataset.local_vars, dtype=np.float32)
#     dataset.labels = np.ascontiguousarray(dataset.labels, dtype=np.float32)
#     dataset.batch_indices = np.ascontiguousarray(dataset.batch_indices, dtype=np.float32)
#     print("MAKING CONTIGUOUS")
#
#     print(dataset)
#
#     data_loaders = SemiSupervisedDataLoaders(dataset, batch_size=batch_size)  # batch_size=512 ?
#     all_weights = np.ones(len(dataset))
#     indices_source = sum((dataset.batch_indices == source).ravel() for source in sources).astype(np.bool)
#     data_loaders['labelled'] = data_loaders(indices=indices_source)
#     indices_target = (dataset.batch_indices == target).ravel().astype(np.bool)
#     data_loaders['unlabelled'] = data_loaders(indices=indices_target)
#
#     #data_loaders['all'] = data_loaders['labelled'] + data_loaders['unlabelled']
#     for key, v in index.items():
#         data_loaders[key] = data_loaders(indices=(dataset.batch_indices == v).ravel())
#
#     data_loaders.loop = ['all', 'labelled']  #   # _weighted
#
#     (X_train, labels_train), = data_loaders.raw_data(data_loaders['labelled'])
#     (X_test, labels_test), = data_loaders.raw_data(data_loaders['unlabelled'])
#     print("MAX Accuracy = ", np.mean([1 if l in np.unique(labels_train) else 0 for l in labels_test]))
#     return dataset, data_loaders, (X_train, labels_train, X_test, labels_test)
#
# #
# names = ['xin', 'segerstolpe', 'muraro', 'baron']
# nb_genes = 300
# batch_size = 128
#
# # dataset_ = GeneExpressionDataset.load_pickle('../scmap/d4-all.pickle')
# # # #
# # all_cell_types = list(
# #     filter(lambda p: p not in ["unknown", "unclassified", "unclassified endocrine", "unclear"], dataset_.cell_types))
# # dataset_.filter_cell_types(all_cell_types)
#
# sources = (0,)
# target = 1
# print("Sources : %s\nTarget : %s" % (' '.join([names[s] for s in sources]), names[target]))
#
# title = "%s -> %s" % (' '.join([names[s] for s in sources]), names[target])
#
# datasets = [XinDataset(), SegerstolpeDataset()]
# dataset_ = GeneExpressionDataset.concat_datasets(*datasets)
#
# dataset, data_loaders, (X_train, labels_train, X_test, labels_test) = split_data(dataset_, sources,
#                                                                                  target,
#                                                                                  nb_genes=nb_genes,
#                                                                                  batch_size=batch_size)
#
#
# #dataset.filter_cell_types()
# vae = VAE(dataset.nb_genes, dataset.n_batches, dataset.n_labels,
#           n_layers=2, n_hidden=256, dropout_rate=0.1)
# # 0.2 in dropout since we move it ?
# infer = VariationalInference(vae, dataset, verbose=True, frequency=50)
# infer.metrics_to_monitor+=['nn_latentspace']
#
# infer.data_loaders = data_loaders
# infer.data_loaders.loop = ['all']
# infer.data_loaders['train'] = data_loaders['all']
# infer.train(300, lr=1e-3)


# infer = SemiSupervisedVariationalInference(svaec, dataset, frequency=10, verbose=True,
#                                            classification_ratio=1,
#                                            n_epochs_classifier=1, lr_classification=1e-3)
#
#
# infer.data_loaders = data_loaders
# infer.classifier_inference.data_loaders['train'] = data_loaders['labelled']
# infer.metrics_to_monitor = ['accuracy', 'll', 'nn_latentspace', 'entropy_batch_mixing']
#
# #infer = adversarial_wrapper(infer, warm_up=200, scale=10)
# infer.train(400, lr=1e-3)
#
# results_svaec[sources][target] = {'acc': infer.accuracy('unlabelled'),
#                                   'bench_acc': infer.benchmark_accuracy('unlabelled'),
#                                   'latent_space_acc': infer.nn_latentspace('unlabelled'),
#                                   'entropy_batch_mixing': infer.entropy_batch_mixing('sequential')}
#
# infer.classifier_inference.train(10)
# results_svaec[sources][target]['after_acc'] = infer.accuracy('unlabelled')
# infer.classifier_inference.train(50)
# results_svaec[sources][target]['really_after_acc'] = infer.accuracy('unlabelled')
# infer.show_t_sne('sequential', color_by='batches and labels', save_name=title + '.svg')
#
# pickle.dump(results_svaec, open(filename_svaec, 'wb'))

# filename_scmap = '4d-scmap-results.pickle'
# if os.path.exists(filename_scmap):
#     results_scmap = pickle.load(open(filename_scmap, 'rb'))
# else:
#     results_scmap = dict()
# print("DOING SCMAP")
# #print(results_scmap)
# nb_genes=300
# scmap = SCMAP()
# for n_features in [300]:#, 500]:
#     scmap.set_parameters(n_features=n_features)
#     if n_features not in results_scmap:
#         results_scmap[n_features] = dict()
#     for i in range(1,4):
#         for sources in list(itertools.combinations(range(0, 4), i)):
#             targets = list(range(4))
#             for source in sources:
#                 targets.remove(source)
#             if sources not in results_scmap[n_features]:
#                 results_scmap[n_features][sources] = dict()
#             for target in targets:
#                 print("Sources : %s\nTarget : %s" % (' '.join([names[s] for s in sources]), names[target]))
#                 title = "%s -> %s" % (' '.join([names[s] for s in sources]), names[target])
#                 dataset, data_loaders, (X_train, labels_train, X_test, labels_test) = split_data(dataset_, sources,
#                                                                                                  target,
#                                                                                                  nb_genes=nb_genes)
#                 scmap.fit_scmap_cluster(X_train, labels_train)
#                 score = scmap.score(X_test, labels_test)
#                 print("Score: ", score)
#                 results_scmap[n_features][sources][target] = score
#
#                 pickle.dump(results_scmap, open(filename_scmap, 'wb'))
#
# print(results_scmap[300])

#
# datasets = [XinDataset(), SegerstolpeDataset(), MuraroDataset(), BaronDataset()]
# filename_scmap = '4d-scmap-results-2.pickle'
# if os.path.exists(filename_scmap):
#     results_scmap = pickle.load(open(filename_scmap, 'rb'))
# else:
#     results_scmap = dict()
# print("DOING SCMAP")
# #print(results_scmap)



# for n_features in [300]:#, 500]:
#     scmap.set_parameters(n_features=n_features)
#     if n_features not in results_scmap:
#         results_scmap[n_features] = dict()
#     for i in range(1,4):
#         for sources in list(itertools.combinations(range(0, 4), i)):
#             targets = list(range(4))
#             for source in sources:
#                 targets.remove(source)
#             if sources not in results_scmap[n_features]:
#                 results_scmap[n_features][sources] = dict()
#             for target in targets:
#                 print("Sources : %s\nTarget : %s" % (' '.join([names[s] for s in sources]), names[target]))
#                 title = "%s -> %s" % (' '.join([names[s] for s in sources]), names[target])
#                 dataset = GeneExpressionDataset.concat_datasets(datasets[target], datasets[sources[0]])
#                 dataset.subsample_genes(n_features)
#                 data_loaders = SemiSupervisedDataLoaders(dataset)
#                 data_loaders['unlabelled'] = data_loaders(indices = (dataset.batch_indices==0).ravel().astype(np.bool))
#                 data_loaders['labelled'] = data_loaders(indices = (dataset.batch_indices==1).ravel().astype(np.bool))
#                 (X_train, labels_train), = data_loaders.raw_data(data_loaders['labelled'])
#                 (X_test, labels_test), = data_loaders.raw_data(data_loaders['unlabelled'])
#                 scmap.fit_scmap_cluster(X_train, labels_train)
#                 score = scmap.score(X_test, labels_test)
#                 print("Score: ", score)
#                 results_scmap[n_features][sources][target] = score
#
#                 pickle.dump(results_scmap, open(filename_scmap, 'wb'))
#
# print(results_scmap[300])
