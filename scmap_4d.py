import itertools
import os
import pickle

from scvi.dataset import *
from scvi.harmonization import SCMAP
from scvi.inference import *
from scvi.models import *

experiment_number = 11
# experiment-10:
# nb_genes = 800
# batch_size=512 : Alternate : lr_classification = 1e-2 : n_layers = 3 : n_hidden_classifier = 512 : nb_genes = 800
# ~ 0.72

# experiment-11:
# nb_genes = 600
# batch_size=512 : Alternate : lr_classification = 1e-2 :  n_layers = 2 : n_hidden_classifier = 256


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

    data_loaders = SemiSupervisedDataLoaders(dataset, batch_size=512)  # batch_size=512 ?
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

    data_loaders.loop = ['all']  # , 'labelled']  # _weighted
    # ['all_weighted', 'labelled_weighted']#'] # maybe labelled_weighted ?

    (X_train, labels_train), = data_loaders.raw_data(data_loaders['labelled'])
    (X_test, labels_test), = data_loaders.raw_data(data_loaders['unlabelled'])
    print("MAX Accuracy = ", np.mean([1 if l in np.unique(labels_train) else 0 for l in labels_test]))
    return dataset, data_loaders, (X_train, labels_train, X_test, labels_test)


names = ['xin', 'segerstolpe', 'muraro', 'baron']
nb_genes = 600

for i in range(3, 0, -1):
    for sources in list(itertools.combinations(range(0, 4), i)):
        targets = list(range(4))
        for source in sources:
            targets.remove(source)
        if sources not in results_svaec:
            results_svaec[sources] = dict()
        for target in targets:
            print("Sources : %s\nTarget : %s" % (' '.join([names[s] for s in sources]), names[target]))

            title = "%s -> %s" % (' '.join([names[s] for s in sources]), names[target])
            dataset, data_loaders, (X_train, labels_train, X_test, labels_test) = split_data(dataset_, sources,
                                                                                             target,
                                                                                             nb_genes=nb_genes)

            svaec = SVAEC(dataset.nb_genes, dataset.n_batches, dataset.n_labels,
                          n_layers=2, dropout_rate=0.1, decoder_scvi_parameters={'n_layers': 2},
                          classifier_parameters={'n_layers': 2, 'n_hidden': 256})

            # infer = SemiSupervisedVariationalInference(svaec, dataset, frequency=10, verbose=True,
            #                                            classification_ratio=1,
            #                                            n_epochs_classifier=1, lr_classification=5*1e-3)

            infer = AlternateSemiSupervisedVariationalInference(svaec, dataset, lr_classification=1e-2,
                                                                n_epochs_classifier=1,
                                                                frequency=100, verbose=True)
            infer.data_loaders = data_loaders
            infer.classifier_inference.data_loaders['train'] = data_loaders['labelled']
            infer.metrics_to_monitor = ['accuracy', 'll', 'nn_latentspace']

            infer.train(800)
            results_svaec[sources][target] = {'acc': infer.accuracy('unlabelled'),
                                              'bench_acc': infer.benchmark_accuracy('unlabelled'),
                                              'latent_space_acc': infer.nn_latentspace('unlabelled'),
                                              'entropy_batch_mixing': infer.entropy_batch_mixing('sequential')}

            infer.classifier_inference.train(10)
            results_svaec[sources][target]['after_acc'] = infer.accuracy('unlabelled')
            infer.classifier_inference.train(50)
            results_svaec[sources][target]['really_after_acc'] = infer.accuracy('unlabelled')
            infer.show_t_sne('sequential', color_by='batches and labels', save_name=title + '.svg')

            pickle.dump(results_svaec, open(filename_svaec, 'wb'))

filename_scmap = '4d-scmap-results.pickle'
if os.path.exists(filename_scmap):
    results_scmap = pickle.load(open(filename_scmap, 'rb'))
else:
    results_scmap = dict()

scmap = SCMAP()
for n_features in [100, 500]:
    scmap.set_parameters(n_features=n_features)
    results_scmap[n_features] = result = dict()
    for i in range(4):
        for sources in list(itertools.combinations(range(0, 4), i)):
            targets = list(range(4))
            for source in sources:
                targets.remove(source)
            if sources not in results_scmap[n_features]:
                results_scmap[n_features][sources] = dict()
            for target in targets:
                print("Sources : %s\nTarget : %s" % (' '.join([names[s] for s in sources]), names[target]))
                title = "%s -> %s" % (' '.join([names[s] for s in sources]), names[target])
                dataset, data_loaders, (X_train, labels_train, X_test, labels_test) = split_data(dataset_, sources,
                                                                                                 target,
                                                                                                 nb_genes=nb_genes)
                scmap.fit_scmap_cluster(X_train, labels_train)
                score = scmap.score(X_test, labels_test)
                print("Score: ", score)
                results_scmap[n_features][sources][target] = score

                pickle.dump(results_scmap, open(filename_scmap, 'wb'))
