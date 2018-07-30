import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from torch.nn import functional as F

from scvi.dataset import CortexDataset
from scvi.dataset.data_loaders import DataLoaders
from scvi.dataset.data_loaders import TrainTestDataLoaders, SemiSupervisedDataLoaders
from scvi.harmonization import SVC
from scvi.metrics.classification import compute_accuracy  # compute_accuracy_svc, compute_accuracy_rf, \
from scvi.metrics.classification import unsupervised_classification_accuracy, compute_accuracy_tuple, \
    compute_predictions
from scvi.metrics.clustering import clustering_scores
from scvi.metrics.clustering import get_latent, entropy_batch_mixing, nn_overlap, select_indices_evenly
from scvi.metrics.differential_expression import de_stats, de_cortex
from scvi.metrics.imputation import imputation, plot_imputation
from scvi.metrics.log_likelihood import compute_log_likelihood
from . import Inference, ClassifierInference

plt.switch_backend('agg')


# Here only what we want to monitor or the essential tasks.
class VariationalInference(Inference):
    r"""The VariationalInference class for the unsupervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SVAEC``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.8``.
        :\**kwargs: Other keywords arguments from the general Inference class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = VariationalInference(gene_dataset, vae, train_size=0.5)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """
    default_metrics_to_monitor = ['ll']

    def __init__(self, model, gene_dataset, train_size=0.8, **kwargs):
        super(VariationalInference, self).__init__(model, gene_dataset, **kwargs)
        self.kl = None
        self.data_loaders = TrainTestDataLoaders(self.gene_dataset, train_size=train_size, use_cuda=self.use_cuda)

    def loss(self, tensors):
        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
        reconst_loss, kl_divergence = self.model(sample_batch, local_l_mean, local_l_var, batch_index)
        loss = torch.mean(reconst_loss + self.kl_weight * kl_divergence)
        return loss

    def on_epoch_begin(self):
        self.kl_weight = self.kl if self.kl is not None else min(1, self.epoch / self.n_epochs) if self.n_epochs else 0

    def ll(self, name, verbose=False):
        ll = compute_log_likelihood(self.model, self.data_loaders[name])
        if verbose:
            print("LL for %s is : %.4f" % (name, ll))
        return ll

    ll.mode = 'min'

    def imputation(self, name, verbose=False, rate=0.1, n_samples=1, n_epochs=1, corruption="uniform",
                   title="Imputation"):
        original_list, imputed_list = imputation(self, name, rate=rate, n_epochs=n_epochs, n_samples=n_samples,
                                                 corruption=corruption)
        # Median of medians for all distances
        median_imputation_score = np.median(np.abs(np.concatenate(original_list) - np.concatenate(imputed_list)))

        # Mean of medians for each cell
        imputation_cells = []
        for original, imputed in zip(original_list, imputed_list):
            has_imputation = len(original) and len(imputed)
            imputation_cells += [np.median(np.abs(original - imputed)) if has_imputation else 0]
        mean_imputation_score = np.mean(imputation_cells)

        if verbose:
            print("Imputation Scores [corruption:%s - rate:%.2f] on  %s after %i:"
                  "\nMedian of Median: %.4f\nMean of Median for each cell: %.4f" %
                  (corruption, rate, name, n_epochs, median_imputation_score, mean_imputation_score))

        plot_imputation(np.concatenate(original_list), np.concatenate(imputed_list), title=title)
        return original_list, imputed_list

    def clustering_scores(self, name, verbose=True, prediction_algorithm='knn'):
        if self.gene_dataset.n_labels > 1:
            latent, _, labels = get_latent(self.model, self.data_loaders[name])

            scores = clustering_scores(latent, labels, prediction_algorithm=prediction_algorithm,
                                       n_labels=self.model.n_labels)
            if verbose:
                print("Clustering Scores for %s with %s:\nSilhouette: %.4f\nNMI: %.4f\nARI: %.4f\nUCA: %.4f" %
                      (name, prediction_algorithm, scores['asw'], scores['nmi'], scores['ari'], scores['uca']))
            return scores

    def nn_overlap_score(self, name='sequential', verbose=True, **kwargs):
        if hasattr(self.gene_dataset, 'adt_expression_clr'):
            assert name == 'sequential'  # only works for the sequential data_loader (mapping indices)
            latent, _, _ = get_latent(self.model, self.data_loaders[name])
            protein_data = self.gene_dataset.adt_expression_clr
            spearman_correlation, fold_enrichment = nn_overlap(latent, protein_data, **kwargs)
            if verbose:
                print("Overlap Scores for %s:\nSpearman Correlation: %.4f\nFold Enrichment: %.4f" %
                      (name, spearman_correlation, fold_enrichment))
            return spearman_correlation, fold_enrichment

    def differential_expression_stats(self, name, *args, **kwargs):
        return de_stats(self.model, self.data_loaders[name], *args, **kwargs)

    def differential_expression(self, name, *args, verbose=False, **kwargs):
        px_scale, all_labels = self.differential_expression_stats(name, *args, **kwargs)

        if type(self.gene_dataset) == CortexDataset:
            if 'use_cuda' in kwargs:
                kwargs.pop('use_cuda')
            de_score = de_cortex(px_scale, all_labels, self.gene_dataset.gene_names, **kwargs)
            if verbose:
                print("DE score for cortex on %s is : %.4f" % (name, de_score))
        return de_score

    differential_expression.mode = 'max'

    def entropy_batch_mixing(self, name, verbose=False, **kwargs):
        if self.gene_dataset.n_batches >= 2:
            print("ignoring %s, doing on sequential" % name)
            name = 'sequential'
            latent, batch_indices, labels = get_latent(self.model, self.data_loaders[name])
            be_score = entropy_batch_mixing(latent, batch_indices, **kwargs)
            if verbose:
                print("Entropy batch mixing %s is : %.4f" % (name, be_score))
            return be_score

    entropy_batch_mixing.mode = 'max'

    def show_t_sne(self, name, n_samples=1000, color_by='', save_name='', uniform=False):
        latent, batch_indices, labels = get_latent(self.model, self.data_loaders[name])
        if uniform:
            idx_t_sne = np.random.permutation(len(latent))[:n_samples] if n_samples else np.arange(len(latent))
        else:
            idx_t_sne = select_indices_evenly(n_samples, batch_indices)
        if latent.shape[1] != 2:
            latent = TSNE().fit_transform(latent[idx_t_sne])
        if not color_by:
            plt.figure(figsize=(10, 10))
            plt.scatter(latent[:, 0], latent[:, 1])
        else:
            batch_indices = batch_indices[idx_t_sne].ravel()
            labels = labels[idx_t_sne].ravel()
            if color_by == 'batches' or color_by == 'labels':
                indices = batch_indices if color_by == 'batches' else labels
                n = self.gene_dataset.n_batches if color_by == 'batches' else self.gene_dataset.n_labels
                if hasattr(self.gene_dataset, 'cell_types') and color_by == 'labels':
                    plt_labels = self.gene_dataset.cell_types
                else:
                    plt_labels = [str(i) for i in range(n)]
                plt.figure(figsize=(10, 10))
                for i, label in zip(range(n), plt_labels):
                    plt.scatter(latent[indices == i, 0], latent[indices == i, 1], label=label)
                plt.legend()
            elif color_by == 'batches and labels':
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                for i in range(self.gene_dataset.n_batches):
                    axes[0].scatter(latent[batch_indices == i, 0], latent[batch_indices == i, 1], label=str(i))
                axes[0].set_title("batch coloring")
                axes[0].axis("off")
                axes[0].legend()

                indices = labels
                if hasattr(self.gene_dataset, 'cell_types'):
                    plt_labels = self.gene_dataset.cell_types
                else:
                    plt_labels = [str(i) for i in range(len(np.unique(indices)))]
                for i, cell_type in zip(range(self.gene_dataset.n_labels), plt_labels):
                    axes[1].scatter(latent[indices == i, 0], latent[indices == i, 1], label=cell_type)
                axes[1].set_title("label coloring")
                axes[1].axis("off")
                axes[1].legend()
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        if save_name:
            plt.savefig(save_name)

    def nn_latentspace(self, name, verbose=True, **kwargs):
        data_train, _, labels_train = self.get_latent('labelled')
        data_test, _, labels_test = self.get_latent('unlabelled')
        nn = KNeighborsClassifier()
        nn.fit(data_train, labels_train)
        score = nn.score(data_test, labels_test)

        print("NN classifier score:", score)
        print("NN classifier tuple:", compute_accuracy_tuple(labels_test, nn.predict(data_test)))
        return score

    def svc_latent_space(self, unit_test=False):
        X_train, _, y_train = self.get_latent('labelled');
        y_train = y_train.ravel()
        X_test, _, y_test = self.get_latent('unlabelled');
        y_test = y_test.ravel()

        svc = SVC()
        svc.fit(X_train, y_train, unit_test=unit_test)
        score = svc.score(X_test, y_test)
        print("SVC latent space score: ",score)
        return score

    def benchmark_ll(self, name, last_n_values=10):
        values = self.history['ll_' + name][-last_n_values:]
        mean = np.mean(values)
        std = 2 * np.sqrt(np.var(values)) / np.sqrt(last_n_values)
        print("LL for %s is : %.4f +- %.4f" % (name, mean, std))
        return mean

    def get_latent(self, name):
        return get_latent(self.model, self.data_loaders[name])


class SemiSupervisedVariationalInference(VariationalInference):
    r"""The abstract SemiSupervisedVariationalInference class for the semi-supervised training of an autoencoder.
    This parent class is inherited to specify the different training schemes for semi-supervised learning
    """
    default_metrics_to_monitor = VariationalInference.default_metrics_to_monitor + ['accuracy']

    def __init__(self, model, gene_dataset, n_labelled_samples_per_class=50, n_epochs_classifier=0,
                 lr_classification=0.1, classification_ratio=100, labels_groups=None, **kwargs):
        super(SemiSupervisedVariationalInference, self).__init__(model, gene_dataset, **kwargs)
        self.labels_groups = labels_groups
        self.n_epochs_classifier = n_epochs_classifier
        self.lr_classification = lr_classification
        self.classification_ratio = classification_ratio
        self.data_loaders = SemiSupervisedDataLoaders(gene_dataset, n_labelled_samples_per_class,
                                                      use_cuda=self.use_cuda)

        self.classifier_inference = ClassifierInference(
            model.classifier, gene_dataset, metrics_to_monitor=[], verbose=True, frequency=0,
            sampling_model=self.model
        )
        self.classifier_inference.data_loaders['train'] = self.data_loaders['labelled']

    def loss(self, tensors_all, tensors_labelled):
        loss = super(SemiSupervisedVariationalInference, self).loss(tensors_all)
        sample_batch, _, _, _, y = tensors_labelled
        classification_loss = F.cross_entropy(self.model.classify(sample_batch), y.view(-1))
        loss += classification_loss * self.classification_ratio
        return loss

    def on_epoch_end(self):
        self.classifier_inference.train(self.n_epochs_classifier, lr=self.lr_classification)
        return super(SemiSupervisedVariationalInference, self).on_epoch_end()

    def accuracy(self, name, verbose=False):
        acc = compute_accuracy(self.model, self.data_loaders[name])
        if verbose:
            print("Acc for %s is : %.4f" % (name, acc))
        return acc

    accuracy.mode = 'max'

    def hierarchical_accuracy(self, name, verbose=False):

        all_y, all_y_pred = compute_predictions(self.model, self.data_loaders[name])
        acc = np.mean(all_y == all_y_pred)

        all_y_groups = np.array([self.model.labels_groups[y] for y in all_y])
        all_y_pred_groups = np.array([self.model.labels_groups[y] for y in all_y_pred])
        h_acc = np.mean(all_y_groups == all_y_pred_groups)

        if verbose:
            print("Acc for %s is : %.4f\nHierarchical Acc for %s is : %.4f\n" % (name, acc, name, h_acc))
        return acc

    accuracy.mode = 'max'

    def benchmark_accuracy(self, name, last_n_values=10, verbose=False):
        values = self.history['accuracy_' + name][-last_n_values:]
        mean = np.mean(values)
        std = 2 * np.sqrt(np.var(values)) / np.sqrt(last_n_values)
        print("Acc for %s is : %.4f +- %.4f" % (name, mean, std))
        return mean

    def unsupervised_accuracy(self, name, verbose=False):
        uca = unsupervised_classification_accuracy(self.model, self.data_loaders[name])[0]
        if verbose:
            print("UCA for %s is : %.4f" % (name, uca))
        return uca

    unsupervised_accuracy.mode = 'max'

    def granular_accuracy(self, name, verbose=False):
        y, y_pred = compute_predictions(self.model, self.data_loaders[name])
        tuple_accuracy = compute_accuracy_tuple(y, y_pred)
        if verbose:
            print(tuple_accuracy.accuracy_classes)
            print(tuple_accuracy.count)
            print("Unweighted : ", tuple_accuracy.unweighted)
        return tuple_accuracy

    def svc_rf(self, **kwargs):
        if 'train' in self.data_loaders:
            raw_data = DataLoaders.raw_data(self.data_loaders['train'], self.data_loaders['test'])
        else:
            raw_data = DataLoaders.raw_data(self.data_loaders['labelled'], self.data_loaders['unlabelled'])
        (data_train, labels_train), (data_test, labels_test) = raw_data
        svc_scores = compute_accuracy_svc(data_train, labels_train, data_test, labels_test, **kwargs)
        rf_scores = compute_accuracy_rf(data_train, labels_train, data_test, labels_test, **kwargs)
        return svc_scores, rf_scores


class AlternateSemiSupervisedVariationalInference(SemiSupervisedVariationalInference):
    r"""The AlternateSemiSupervisedVariationalInference class for the semi-supervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAEC``, ``SVAEC``, ...
        :gene_dataset: A gene_dataset instance with pre-annotations like ``CortexDataset()``
        :n_labelled_samples_per_class: The number of labelled training samples per class. Default: ``50``.
        :**kwargs: Other keywords arguments from the general Inference class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> svaec = SVAEC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = AlternateSemiSupervisedVariationalInference(gene_dataset, svaec, n_labelled_samples_per_class=10)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """

    def __init__(self, *args, **kwargs):
        super(AlternateSemiSupervisedVariationalInference, self).__init__(*args, **kwargs)
        self.data_loaders.loop = ['all']

    def loss(self, tensors_all):
        return VariationalInference.loss(self, tensors_all)


class JointSemiSupervisedVariationalInference(SemiSupervisedVariationalInference):
    r"""The JointSemiSupervisedVariationalInference class for the semi-supervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAEC``, ``SVAEC``, ...
        :gene_dataset: A gene_dataset instance with pre-annotations like ``CortexDataset()``
        :n_labelled_samples_per_class: The number of labelled training samples per class. Default: ``50``.
        :**kwargs: Other keywords arguments from the general Inference class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> svaec = SVAEC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = JointSemiSupervisedVariationalInference(gene_dataset, svaec, n_labelled_samples_per_class=10)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """

    def __init__(self, *args, **kwargs):
        kwargs['n_epochs_classifier'] = 0
        super(JointSemiSupervisedVariationalInference, self).__init__(*args, **kwargs)
