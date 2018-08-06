import numpy as np

from scvi.dataset import *
from scvi.dataset.BICCN import order
from scvi.inference import *
from scvi.models import *


def run_scanvi(source_datasets, target_dataset, params, max_acc=True):
    if type(source_datasets) not in [tuple, list]:
        source_datasets = (source_datasets,)

    title = "%s->%s" % (' '.join([s.__class__.__name__ for s in source_datasets]), target_dataset.__class__.__name__)
    print(title)
    results = dict()

    dataset = GeneExpressionDataset.concat_datasets(target_dataset, *source_datasets)
    dataset.subsample_genes(new_n_genes=params['nb_genes'])
    if not max_acc:  # Then it is the Macosko-Regev dataset
        for group in order:
            dataset.merge_cell_types(group, group[0])

    data_loaders = SemiSupervisedDataLoaders(dataset, batch_size=params['batch_size'])
    data_loaders['unlabelled'] = data_loaders(indices=(dataset.batch_indices == 0).ravel().astype(np.bool))
    data_loaders['labelled'] = data_loaders['train'] = \
        data_loaders(indices=(dataset.batch_indices > 0).ravel().astype(np.bool))
    if max_acc:
        (_, labels_train), = data_loaders.raw_data(data_loaders['labelled'])
        (_, labels_test), = data_loaders.raw_data(data_loaders['unlabelled'])
        max_acc = np.mean([1 if l in np.unique(labels_train) else 0 for l in labels_test])
        print("Maximum Accuracy : ", max_acc)
        results.update({'max_acc': max_acc})

    # ~ equivalent to a warm-up for the classification
    vae = VAE(dataset.nb_genes, dataset.n_batches, dataset.n_labels,
              n_layers=params['n_layers'], n_hidden=params['n_hidden'], dropout_rate=0.1)
    infer = VariationalInference(vae, dataset, weight_decay=params['weight_decay'])

    infer.data_loaders = data_loaders
    infer.data_loaders.loop = ['all']
    infer.data_loaders['train'] = data_loaders['all']
    infer.train(params['n_epoch_train_vae'], lr=params['lr_train_vae'])

    results.update({
        'vae_latent_space_acc': infer.nn_latentspace('unlabelled'),
        'vae_entropy_batch_mixing': infer.entropy_batch_mixing('sequential'),
        'vae_clustering_scores': infer.clustering_scores('unlabelled')
    })

    scanvi = SCANVI(dataset.nb_genes, dataset.n_batches, dataset.n_labels,
                    n_layers=params['n_layers'], n_hidden=params['n_hidden'], dropout_rate=0.1,
                    classifier_parameters=params['classifier_parameters'])
    scanvi.load_state_dict(vae.state_dict(), strict=False)
    infer_scanvi = SemiSupervisedVariationalInference(
        scanvi, dataset, frequency=10, verbose=False, classification_ratio=params['classification_ratio'],
        n_epochs_classifier=1, lr_classification=params['lr_classification']
    )

    infer_scanvi.data_loaders = data_loaders
    data_loaders.loop = ['all', 'labelled']
    infer_scanvi.classifier_inference.data_loaders['labelled'] = data_loaders['labelled']
    infer_scanvi.classifier_inference.data_loaders.loop = ['labelled']

    print(infer_scanvi.nn_latentspace('sequential'))
    infer_scanvi.train(params['n_epoch_train_scanvi'], lr=params['lr_train_scanvi'])

    results.update({
        'acc': infer_scanvi.accuracy('unlabelled'),
        'bench_acc': infer_scanvi.benchmark_accuracy('unlabelled'),
        'latent_space_acc': infer_scanvi.nn_latentspace('unlabelled'),
        'entropy_batch_mixing': infer_scanvi.entropy_batch_mixing('sequential'),
        'scanvi_clustering_scores': infer_scanvi.clustering_scores('unlabelled')
    })

    infer_scanvi.classifier_inference.train(10, batch_norm=False)
    results['after_acc'] = infer_scanvi.accuracy('unlabelled')
    infer_scanvi.classifier_inference.train(50, batch_norm=False)
    results['really_after_acc'] = infer_scanvi.accuracy('unlabelled')
    if params['save_t_sne_folder'] is not None:
        infer_scanvi.show_t_sne('sequential', color_by='batches and labels',
                                save_name=params['save_t_sne_folder'] + title + '.svg')

    return results
