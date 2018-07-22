from scvi.dataset import CortexDataset
from scvi.inference import VariationalInference, AlternateSemiSupervisedVariationalInference, \
    JointSemiSupervisedVariationalInference
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
    def annotation_benchmarks(self, dataset, n_epochs=80, n_labelled_samples_per_class=30, frequency=10, verbose=True):
        self.vae = VAE(dataset.nb_genes, dataset.n_batches, dataset.n_labels)
        self.svaec = SVAEC(dataset.nb_genes, 0, dataset.n_labels)
        self.svaec_logreg = SVAEC(dataset.nb_genes, 0, dataset.n_labels, logreg_classifier=True)

        self.infer_svaec_joint = JointSemiSupervisedVariationalInference(
            self.svaec, dataset, n_labelled_samples_per_class=n_labelled_samples_per_class, verbose=verbose,
            frequency=frequency
        )
        self.infer_svaec_alternate = AlternateSemiSupervisedVariationalInference(
            self.svaec_logreg, dataset, n_labelled_samples_per_class=n_labelled_samples_per_class, verbose=True,
            frequency=frequency
        )

        self.infer_vae = VariationalInference(self.vae, dataset)
        self.infer_vae.data_loaders = self.infer_svaec_alternate.data_loaders
        self.infer_vae.train(n_epochs=n_epochs)

        self.infer_svaec_alternate.train(n_epochs=n_epochs)
        self.infer_svaec_joint.train(n_epochs=n_epochs)

        print("VAE")
        self.infer_vae.clustering_scores('sequential')
        print("JOINT")
        self.infer_svaec_joint.clustering_scores('sequential')
        print("ALTERNATE")
        self.infer_svaec_alternate.clustering_scores('sequential')


        svc_scores, rf_scores = self.infer_svaec_joint.svc_rf_latent_space(unit_test=(n_epochs==1))
        print('SVC Joint:', svc_scores[1].unweighted)
        print('RF Joint:', rf_scores[1].unweighted)

        svc_scores, rf_scores = self.infer_vae.svc_rf_latent_space(unit_test=(n_epochs==1))
        print('VAE SVC :', svc_scores[1].unweighted)
        print('VAE RF :', rf_scores[1].unweighted)
