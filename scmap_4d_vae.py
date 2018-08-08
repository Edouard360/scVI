import itertools
import os
import pickle
import copy
import numpy as np

from scvi.dataset import *
from scvi.harmonization import SCMAP
from scvi.inference import *
from scvi.models import *
import torch
experiment_number = 15



# experiment-10:
# nb_genes = 800
# batch_size=512 : Alternate : lr_classification = 1e-2 : n_layers = 3 : n_hidden_classifier = 512 : nb_genes = 800
# ~ 0.72

# experiment-11:
# nb_genes = 600
# batch_size=512 : Alternate : lr_classification = 1e-2 :  n_layers = 2 : n_hidden_classifier = 256

# experiment-12:
# nb_genes = 600
# batch_size=128 : Joint : lr_classification = 5*1e-3 :  n_layers = 2 : n_hidden_classifier = 256

# experiment-13:
# nb_genes = 1000
# batch_size=128 : General : classification_ratio=1, lr_classification = 5*1e-3 :
# n_layers = 1 : n_layers_classifier = 2 : n_hidden_classifier = 256

# experiment-14:
#  lr=1e-4 : no batch mixing at all !

# experiment-15:
#  lr=1e-2 : batch mixing ? - seems to work on 1 -> 0

filename_vae = 'scmap_experiments/4d-vae-2-results.pickle'

if os.path.exists(filename_vae):
    results_vae = pickle.load(open(filename_vae, 'rb'))
else:
    results_vae = dict()

dataset_ = GeneExpressionDataset.load_pickle('../scmap/d4-all.pickle')

all_cell_types = list(
    filter(lambda p: p not in ["unknown", "unclassified", "unclassified endocrine", "unclear"], dataset_.cell_types))
dataset_.filter_cell_types(all_cell_types)
index = {'xin': 0, 'segerstolpe': 1, 'muraro': 2, 'baron': 3}


def split_data(original_dataset, sources, target, nb_genes=1500, batch_size=128):
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

    data_loaders = SemiSupervisedDataLoaders(dataset, batch_size=batch_size)  # batch_size=512 ?
    all_weights = np.ones(len(dataset))
    indices_source = sum((dataset.batch_indices == source).ravel() for source in sources).astype(np.bool)
    data_loaders['labelled'] = data_loaders(indices=indices_source)
    indices_target = (dataset.batch_indices == target).ravel().astype(np.bool)
    data_loaders['unlabelled'] = data_loaders(indices=indices_target)

    data_loaders['all'] = data_loaders['labelled'] + data_loaders['unlabelled']
    for key, v in index.items():
        data_loaders[key] = data_loaders(indices=(dataset.batch_indices == v).ravel())

    data_loaders.loop = ['all', 'labelled']

    (X_train, labels_train), = data_loaders.raw_data(data_loaders['labelled'])
    (X_test, labels_test), = data_loaders.raw_data(data_loaders['unlabelled'])
    max_acc = np.mean([1 if l in np.unique(labels_train) else 0 for l in labels_test])
    print("MAX Accuracy = ", max_acc)
    return dataset, data_loaders, (X_train, labels_train, X_test, labels_test), max_acc


filename_svaec = 'scmap_experiments/4d-svaec-results-best.pickle'

if os.path.exists(filename_svaec):
    results_svaec = pickle.load(open(filename_svaec, 'rb'))
else:
    results_svaec = dict()

names = ['xin', 'segerstolpe', 'muraro', 'baron']
nb_genes = 300
batch_size = 128

n_epoch_train_vae = 1200
n_epoch_train_svaec = 200

for i in range(3, 1, -1): # Just 3 and 2
    for sources in list(itertools.combinations(range(0, 4), i)):
        targets = list(range(4))
        for source in sources:
            targets.remove(source)
        if sources not in results_vae:
            results_vae[sources] = dict()
        if sources not in results_svaec:
            results_svaec[sources] = dict()
        for target in targets:
            print("Sources : %s\nTarget : %s" % (' '.join([names[s] for s in sources]), names[target]))

            title = "%s->%s" % (' '.join([names[s] for s in sources]), names[target])
            dataset, data_loaders, (X_train, labels_train, X_test, labels_test), max_acc = split_data(dataset_, sources,
                                                                                             target,
                                                                                             nb_genes=nb_genes,
                                                                                             batch_size=batch_size)

            vae = VAE(dataset.nb_genes, dataset.n_batches, dataset.n_labels,
                          n_layers=2, n_hidden=256, dropout_rate=0.1)
            # 0.2 in dropout since we move it ?
            infer = VariationalInference(vae, dataset)

            infer.data_loaders = data_loaders
            infer.data_loaders.loop = ['all']
            infer.data_loaders['train'] = data_loaders['all']
            infer.train(n_epoch_train_vae, lr=1e-3)

            vae_state_file = 'scmap_experiments/state_dict/'+title
            torch.save(vae.state_dict(), vae_state_file)

            results_vae[sources][target] = {
                'latent_space_acc': infer.nn_latentspace('unlabelled'),
                'entropy_batch_mixing': infer.entropy_batch_mixing('sequential')
            }
            pickle.dump(results_vae, open(filename_vae, 'wb'))


            svaec = SCANVI(dataset.nb_genes, dataset.n_batches, dataset.n_labels,  #dispersion='gene-batch',
                          n_layers=2, n_hidden=256, dropout_rate=0.1)
            svaec.eval()
            infer_svaec = SemiSupervisedVariationalInference(svaec, dataset, frequency=10, verbose=True,
                                                             classification_ratio=5,
                                                        n_epochs_classifier=1, lr_classification=5*1e-4)
            svaec.load_state_dict(torch.load(open(vae_state_file, 'rb')), strict=False)
            infer_svaec.data_loaders = data_loaders
            infer_svaec.classifier_inference.data_loaders['labelled'] = data_loaders['labelled']
            infer_svaec.classifier_inference.data_loaders.loop = ['labelled']
            infer_svaec.classifier_inference.data_loaders['train'] = data_loaders['labelled']
            data_loaders.loop = ['all', 'labelled']
            print(infer_svaec.nn_latentspace('sequential'))
            #, logreg_classifier=True)#decoder_scvi_parameters={'n_layers': 3, 'n_hidden': 64},
            #classifier_parameters={'n_layers': 2, 'n_hidden': 128, 'dropout_rate':0.2}

            #infer_svaec = adversarial_wrapper(infer_svaec, warm_up=0, scale=50)
            infer_svaec.train(n_epoch_train_svaec, lr=1e-4, weight_decay=1e-6)


            results_svaec[sources][target] = {'acc': infer_svaec.accuracy('unlabelled'),
                                              'bench_acc': infer_svaec.benchmark_accuracy('unlabelled'),
                                              'latent_space_acc': infer_svaec.nn_latentspace('unlabelled'),
                                              'entropy_batch_mixing': infer_svaec.entropy_batch_mixing('sequential'),
                                              'max_acc':max_acc}

            infer_svaec.classifier_inference.train(10, lr=1e-4, batch_norm=False)
            results_svaec[sources][target]['after_acc'] = infer_svaec.accuracy('unlabelled')
            infer_svaec.classifier_inference.train(50, lr=1e-4, batch_norm=False)
            results_svaec[sources][target]['really_after_acc'] = infer_svaec.accuracy('unlabelled')

            pickle.dump(results_svaec, open(filename_svaec, 'wb'))
            # infer_svaec = AlternateSemiSupervisedVariationalInference(svaec, dataset, frequency=10, verbose=True, classification_ratio=0,
            #                                             n_epochs_classifier=1, lr_classification=5*1e-4)#,#classification_ratio=10,

            # infer_svaec = VariationalInference(svaec, dataset, frequency=10, verbose=True,#,
            #                                            n_epochs_classifier=1, lr_classification=5*1e-4)



            #state_dict = vae.state_dict()



            #infer_svaec.classifier_inference.train(50, lr=5*1e-4, weight_decay=0) #  0.936
            # print(infer_svaec.accuracy('unlabelled'))
            #svaec.load_state_dict(torch.load(open('scmap_experiments/state_dict', 'rb')), strict=False)

            # print("OK")
            # print("THEN")
            #
            #infer_svaec.train(200, lr=0, weight_decay=0) # this will train at lr_classification=5*1e-4, with weight decay !!!
            #  # WARNING THERE IS A DIFFERENCE !
            #
            # raise
            #raise



# from scvi.models.classifier import Classifier
#  cls = Classifier(n_input=10, n_labels=dataset.n_labels, n_layers=2, n_hidden=128, dropout_rate=0.2)
#  infer_cls = ClassifierInference(cls, dataset, sampling_model=svaec, verbose=True, frequency=10)
#
#  infer_cls.data_loaders['labelled'] = data_loaders['labelled']
#  infer_cls.data_loaders['unlabelled'] = data_loaders['unlabelled']
#  infer_cls.data_loaders.to_monitor = ['labelled', 'unlabelled']
#  infer_cls.data_loaders.loop = ['labelled']
#  infer_cls.train(200, lr=5 * 1e-4, weight_decay=0)
