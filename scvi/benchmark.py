from scvi.benchmarks import SCMAP
from scvi.dataset import CortexDataset, GeneExpressionDataset, SemiSupervisedDataLoaders
from scvi.inference import VariationalInference, SemiSupervisedVariationalInference
from scvi.models import VAE, SVAEC


def cortex_benchmark(n_epochs=250, use_cuda=True, unit_test=False):
    cortex_dataset = CortexDataset()
    vae = VAE(cortex_dataset.nb_genes)
    infer_cortex_vae = VariationalInference(vae, cortex_dataset, use_cuda=use_cuda)
    infer_cortex_vae.train(n_epochs=n_epochs)

    infer_cortex_vae.ll('test')  # assert ~ 1200
    infer_cortex_vae.differential_expression('test')
    infer_cortex_vae.imputation('test', rate=0.1)  # assert ~ 2.3
    n_samples = 1000 if not unit_test else 10
    infer_cortex_vae.show_t_sne('test', n_samples=n_samples)
    return infer_cortex_vae


def benchmark(dataset, n_epochs=250, use_cuda=True):
    vae = VAE(dataset.nb_genes, n_batch=dataset.n_batches)
    infer = VariationalInference(vae, dataset, use_cuda=use_cuda)
    infer.train(n_epochs=n_epochs)
    infer.ll('test')
    infer.imputation('test', rate=0.1)  # assert ~ 2.1
    return infer


def harmonization_benchmarks(n_epochs=1, use_cuda=True):
    # retina_benchmark(n_epochs=n_epochs)
    pass


def annotation_benchmarks(n_epochs=1, use_cuda=True):
    # some cortex annotation benchmark
    pass


def all_benchmarks(n_epochs=250, use_cuda=True, unit_test=False):
    cortex_benchmark(n_epochs=n_epochs, use_cuda=use_cuda, unit_test=unit_test)

    harmonization_benchmarks(n_epochs=n_epochs, use_cuda=use_cuda)
    annotation_benchmarks(n_epochs=n_epochs, use_cuda=use_cuda)


class Benchmark:
    def self_prediction_benchmarks(self, n_epochs=80, frequency=10, max_iter=10000):
        '''
        1. For many splits
        running_time
        uca, ari, nmi, asw
        LL
        200 epochs

        2. For last split
        PCA + T-SNE
        t_sne_plots is save_name = .svg pour VAE + SVAEC



        '''

        dataset_splits = [
            (CortexDataset, [5, 10, 20]),
            # (PurePBMC, [10,20,100])#,1000])
            # (SyntheticSimilar, [5,10,20,100]),

        ]
        classifier_parameters = {
            'n_hidden': 128,
            'n_layers': 1,
            'dropout_rate': 0.1
        }
        print(classifier_parameters)
        for Dataset, splits in dataset_splits:
            dataset = Dataset()
            last_split = splits[-1]
            for split in splits:
                print("SVAEC")  # First do this for agreement over dataloaders
                model = SVAEC(dataset.nb_genes, 0, dataset.n_labels, classifier_parameters=classifier_parameters)
                infer = SemiSupervisedVariationalInference(
                    model, dataset, n_labelled_samples_per_class=split, n_epochs_classifier=10,
                    lr_classification=1e-2, classification_ratio=100,
                    frequency=frequency, verbose=True
                )
                infer.train(n_epochs=n_epochs)
                infer.benchmark_accuracy('unlabelled', last_n_values=10)
                infer.ll('sequential', verbose=True)
                svc_scores, rf_scores = infer.svc_rf(max_iter=max_iter)
                print('SVC Joint:', svc_scores[1].unweighted)
                print('RF Joint:', rf_scores[1].unweighted)
                svc_scores, rf_scores = infer.svc_rf_latent_space(max_iter=max_iter)
                print('SVC Joint:', svc_scores[1].unweighted)
                print('RF Joint:', rf_scores[1].unweighted)
                print("Clustering score SVAEC/Joint")
                infer.clustering_scores('sequential')
                if split == last_split:
                    infer.show_t_sne('sequential', color_by='labels', save_name='tsne.svg')
                    # data_loaders_svaec = infer.data_loaders

                    # print("VAE")
                    # model = VAE(dataset.nb_genes, 0, dataset.n_labels)
                    # infer = VariationalInference(
                    #     model, dataset, frequency=frequency
                    # )
                    # infer.data_loaders['train'] = infer.data_loaders(shuffle=True)  # use all to compare to semi-supervised
                    # infer.data_loaders['labelled'] = data_loaders_svaec['labelled']
                    # infer.data_loaders['unlabelled'] = data_loaders_svaec['unlabelled']
                    # infer.train(n_epochs=n_epochs)
                    # infer.ll('sequential', verbose=True)
                    #
                    # svc_scores, rf_scores = infer.svc_rf_latent_space(max_iter=max_iter)
                    # print('SVC Joint:', svc_scores[1].unweighted)
                    # print('RF Joint:', rf_scores[1].unweighted)
                    # print("Clustering score VAE")
                    # infer.clustering_scores('sequential')
                    # if split == last_split:
                    #     infer.show_t_sne('sequential', color_by='labels', save_name='tsne.svg')

    def scmap_cross_benchmarks(self, n_epochs):

        ''' This is for section 2:
        :return:
        '''
        # (source, target) [or (reference, projection)]
        # can do multiple source one target in our case.
        scmap = SCMAP()
        dataset_xin = scmap.create_dataset("../scmap/xin/xin.rds")
        cell_types_xin = list(filter(lambda p: ".contaminated" not in p, dataset_xin.cell_types))
        dataset_xin.filter_cell_types(cell_types_xin)

        dataset_segerstolpe = scmap.create_dataset("../scmap/segerstolpe/segerstolpe.rds")
        cell_types_segerstolpe = list(filter(lambda p: p != "not applicable", dataset_segerstolpe.cell_types))
        dataset_segerstolpe.filter_cell_types(cell_types_segerstolpe)

        dataset_muraro = scmap.create_dataset("../scmap/muraro/muraro.rds")
        dataset_baron = scmap.create_dataset("../scmap/baron-human/baron-human.rds")

        concatenated = GeneExpressionDataset.concat_datasets(dataset_xin, dataset_segerstolpe, dataset_muraro,
                                                             dataset_baron, on='gene_symbols')  #
        concatenated.subsample_genes(subset_genes=(concatenated.X.max(axis=0) <= 2500).ravel())
        all_cell_types = list(
            filter(lambda p: p not in ["unknown", "unclassified", "unclear", "unclassified endocrine"],
                   concatenated.cell_types))
        concatenated.filter_cell_types(all_cell_types)

        data_loaders = SemiSupervisedDataLoaders(concatenated, n_labelled_samples_per_class=10)
        data_loaders['xin'] = data_loaders(indices=(concatenated.batch_indices == 0).ravel())
        data_loaders['segerstolpe'] = data_loaders(indices=(concatenated.batch_indices == 1).ravel())
        data_loaders['muraro'] = data_loaders(indices=(concatenated.batch_indices == 2).ravel())
        data_loaders['baron'] = data_loaders(indices=(concatenated.batch_indices == 3).ravel())

        data_loaders.to_monitor = ['xin', 'segerstolpe', 'muraro', 'sequential']
        data_loaders.data_loaders_loop = ['all', 'labelled']

        sources_target = [
            ('segerstolpe', 'xin'),
            ('xin', 'segerstolpe'),
            (['segerstolpe','muraro'],'xin')
        ]
        for sources, target in sources_target:
            if type(sources) is str:
                sources = [sources]
            data_loaders['labelled'] = sum(data_loaders[source] for source in sources)
            data_loaders['all'] = data_loaders['labelled']+data_loaders[target]
            data_loaders.to_monitor = [target]

            (X_train, labels_train), = data_loaders.raw_data(data_loaders['labelled'])
            (X_test, labels_test), = data_loaders.raw_data(data_loaders[target])
            scmap.create_sce_object(X_train, concatenated.gene_names, labels_train, 'd_train')
            scmap.create_sce_object(X_test, concatenated.gene_names, labels_test, 'd_test')
            print(len(labels_train))

            labels_pred = scmap.scmap_cluster(
                'd_train', 'd_test', threshold=0, n_features=500
            )
            print("Accuracy result scmap_cluster:", scmap.accuracy_tuple(labels_test, labels_pred))

            labels_pred = scmap.scmap_cell(
                'd_train', 'd_test', w=10, n_features=500, threshold=0
            )
            print("Accuracy result scmap_cell:", scmap.accuracy_tuple(labels_test, labels_pred))

            svaec = SVAEC(concatenated.nb_genes, concatenated.n_batches, concatenated.n_labels)

            infer = SemiSupervisedVariationalInference(
                svaec, concatenated, frequency=10, verbose=True,
            )
            infer.data_loaders = data_loaders
            infer.train(n_epochs=n_epochs)
            infer.benchmark_accuracy(target, last_n_values=10)
