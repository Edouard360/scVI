import numpy as np

from scvi.dataset import *
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
    if not max_acc:
        preprocess(dataset)

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
    vae = VAE(dataset.nb_genes, dataset.n_batches, dataset.n_labels, #dispersion='gene-batch',
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


def preprocess(dataset):
    key_color_order = [['Pvalb low', 'Pvalb', 'Pvalb 1', 'Pvalb 2'],
                       ['Pvalb Ex_1', 'Pvalb Ex_2', 'Pvalb Ex'],
                       ['Pvalb Astro_1', 'Pvalb Astro_2'],
                       ['L2/3 IT Astro', 'L2/3 IT Macc1', 'L2/3 IT Sla_Astro', 'L2/3 IT', 'L2/3 IT Sla', 'L2/3 IT Sla_Inh'],
                       ['Sst Tac2', 'Sst Myh8', 'Sst Etv1', 'Sst Chodl', 'Sst'],
                       ['L5 PT_2', 'L5 PT IT', 'L5 PT_1'],
                       ['L5 IT Tcap_1_3', 'L5 IT Tcap_2', 'L5 IT Tcap_Astro', 'L5 IT Tcap_1', 'L5 IT Tcap_L2/3',
                        'L5 IT Tcap_Foxp2', 'L5 IT Tcap_3'],
                       ['L5 IT Aldh1a7_2', 'L5 IT Aldh1a7', 'L5 IT Aldh1a7_1'],
                       ['L5 NP', 'L5 NP Slc17a8'],
                       ['L6 IT Car3', 'L6 CT Olig', 'L6 IT Maf', 'L6 IT Ntn5 Mgp', 'L6 IT Ntn5 Inpp4b'],
                       ['L6 CT Nxph2', 'L6 CT Astro', 'L6 CT', 'L6 CT Grp'],
                       ['L6b', 'L6b F2r'],
                       ['Lamp5 Sncg', 'Lamp5 Egln3', 'Lamp5 Slc35d3'],
                       ['Vip Rspo4', 'Vip Serpinf1', 'Vip'],
                       ['Astro Ex', 'Astro Aqp4'],
                       ['OPC Pdgfra'],
                       ['VLMC Osr1'],
                       ['Oligo Enpp6_1', 'Oligo Enpp6_2', 'Oligo Opalin'],
                       ['Sncg Ptprk'],
                       ['Endo Slc38a5', 'Endo Slc38a5_Peri_2', 'Endo Slc38a5_Peri_1']]

    key_color_order = [[key for key in key_color_group] for key_color_group in key_color_order]
    ravel = [key for key_color in key_color_order for key in key_color]
    labels_group_array = [len(key) * [group] for group, key in enumerate(key_color_order)]
    labels_groups = []
    for group in labels_group_array:
        labels_groups += group

    for group in key_color_order:
        dataset.merge_cell_types(group, group[0])

# Part I - predicting one dataset from another

# Synthetic UMI-Non UMI
'''
results_umi_nonumi = pd.DataFrame(index=['UMI SCANVI', 'UMI SCMAP', 'NON-UMI SCANVI', 'NON-UMI SCMAP'],
                                  columns=['UMI', 'NON-UMI'])
synthetic_umi = SyntheticUMI()
synthetic_nonumi = SyntheticNONUMI()

params.update({
    'n_epoch_train_vae': 1,  # 100,
    'nb_genes': 100,  # 1000,
    'n_epoch_train_scanvi': 1,  # 200,
    'lr_train_scanvi': 1e-4
})

umi_to_nonumi = run_scanvi(synthetic_umi, synthetic_nonumi, params)
nonumi_to_umi = run_scanvi(synthetic_nonumi, synthetic_umi, params)

results_umi_nonumi['UMI SCANVI']['NON-UMI'] = umi_to_nonumi['acc']
results_umi_nonumi['NON-UMI SCANVI']['UMI'] = nonumi_to_umi['acc']

results_umi_nonumi

# Macosko Regev

results_macosko_regev = pd.DataFrame(index=['Macosko SCANVI', 'Macosko SCMAP', 'Regev SCANVI', 'Regev SCMAP'],
                                   columns=['Macosko', 'Regev'])
macosko = MacoskoDataset()
regev = RegevDataset()

macosko_to_regev = run_scanvi(macosko, regev, params)
regev_to_macosko = run_scanvi(regev, macosko, params)

results_macosko_regev['Macosko SCANVI']['Regev'] = macosko_to_regev['acc']
results_macosko_regev['Regev SCANVI']['Macosko'] = regev_to_macosko['acc']

results_macosko_regev['Macosko SCMAP']['Regev'] = 0.911
results_macosko_regev['Regev SCMAP']['Macosko'] = 0.870

results_macosko_regev

# scmap datasets

params.update({
    'n_epoch_train_vae': 1200,
    'nb_genes': 300,
    'n_epoch_train_scanvi': 200,
    'lr_train_scanvi': 5 * 1e-4
})

datasets = [XinDataset(), SegerstolpeDataset(), MuraroDataset(), BaronDataset()]
for dataset in datasets:
    dataset.subsample_genes(subset_genes=(dataset.X.max(axis=0) <= 2500).ravel())

results = dict()

for source in range(4):
    targets = list(range(4))
    targets.remove(source)
    if source not in results:
        results[source] = dict()
    for target in targets:
        results[source][target] = run_scanvi(datasets[target], datasets[source], params)


sources = (0, 1)
target = 2

results[sources] = dict()
results_0_1_to_2 = run_scanvi(datasets[target], [datasets[source] for source in sources], params)
results[sources][target] = results_0_1_to_2
'''
