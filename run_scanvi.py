import numpy as np

from scvi.dataset import *
from scvi.inference import *
from scvi.models import *

params = {
    'n_epoch_train_vae': 1200,
    'n_layers': 2,
    'n_hidden': 256,
    'nb_genes': 300,
    'classifier_parameters': dict(),
    'n_epoch_train_scanvi': 200,
    'lr_train_scanvi': 1e-4,
    'classification_ratio':10,
    'lr_classification':1e-3
}


def run_scanvi(target_dataset, source_datasets, params):
    if type(source_datasets) is not tuple:
        source_datasets = (source_datasets,)

    # ~ equivalent to a warm-up for the classification
    # str(dataset) for dataset in source_datasets

    title = "%s->%s" % (' '.join([s.__class__.__name__ for s in source_datasets]), target_dataset.__class__.__name__)
    print(title)
    results = dict()

    dataset = GeneExpressionDataset.concat_datasets(target_dataset, *source_datasets)
    dataset.subsample_genes(new_n_genes=params['nb_genes'])

    data_loaders = SemiSupervisedDataLoaders(dataset)
    data_loaders['unlabelled'] = data_loaders(indices=(dataset.batch_indices == 0).ravel().astype(np.bool))
    data_loaders['labelled'] = data_loaders['train'] = \
        data_loaders(indices=(dataset.batch_indices > 0).ravel().astype(np.bool))

    (_, labels_train), = data_loaders.raw_data(data_loaders['labelled'])
    (_, labels_test), = data_loaders.raw_data(data_loaders['unlabelled'])
    max_acc = np.mean([1 if l in np.unique(labels_train) else 0 for l in labels_test])
    print("Maximum Accuracy : ", max_acc)
    results.update({'max_acc': max_acc})

    vae = VAE(dataset.nb_genes, dataset.n_batches, dataset.n_labels,
              n_layers=params['n_layers'], n_hidden=params['n_hidden'], dropout_rate=0.1)
    infer = VariationalInference(vae, dataset)

    infer.data_loaders = data_loaders
    infer.data_loaders.loop = ['all']
    infer.data_loaders['train'] = data_loaders['all']
    infer.train(params['n_epoch_train_vae'], lr=1e-3)

    results.update({
        'vae_latent_space_acc': infer.nn_latentspace('unlabelled'),
        'vae_entropy_batch_mixing': infer.entropy_batch_mixing('sequential')
    })

    svaec = SVAEC(dataset.nb_genes, dataset.n_batches, dataset.n_labels,
                  n_layers=params['n_layers'], n_hidden=params['n_hidden'], dropout_rate=0.1,
                  classifier_parameters=params['classifier_parameters'])
    svaec.load_state_dict(vae.state_dict(), strict=False)
    infer_svaec = SemiSupervisedVariationalInference(
        svaec, dataset, frequency=10, verbose=False, classification_ratio=params['classification_ratio'],
        n_epochs_classifier=1, lr_classification=params['lr_classification']
    )

    infer_svaec.data_loaders = data_loaders
    data_loaders.loop = ['all', 'labelled']
    infer_svaec.classifier_inference.data_loaders['labelled'] = data_loaders['labelled']
    infer_svaec.classifier_inference.data_loaders.loop = ['labelled']

    print(infer_svaec.nn_latentspace('sequential'))
    infer_svaec.train(params['n_epoch_train_scanvi'], lr=params['lr_train_scanvi'])

    results.update({
        'acc': infer_svaec.accuracy('unlabelled'),
        'bench_acc': infer_svaec.benchmark_accuracy('unlabelled'),
        'latent_space_acc': infer_svaec.nn_latentspace('unlabelled'),
        'entropy_batch_mixing': infer_svaec.entropy_batch_mixing('sequential')
    })

    infer_svaec.classifier_inference.train(10)
    results['after_acc'] = infer_svaec.accuracy('unlabelled')
    infer_svaec.classifier_inference.train(50)
    results['really_after_acc'] = infer_svaec.accuracy('unlabelled')
    infer_svaec.show_t_sne('sequential', color_by='batches and labels', save_name=title + '.svg')

    return results
